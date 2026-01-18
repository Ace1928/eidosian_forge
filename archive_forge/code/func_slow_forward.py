import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import (
from .configuration_jamba import JambaConfig
def slow_forward(self, input_states, cache_params: HybridMambaAttentionDynamicCache=None):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    projected_states = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)
    use_cache = isinstance(cache_params, HybridMambaAttentionDynamicCache)
    if use_cache and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
        if self.training:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        else:
            ssm_state = cache_params.ssm_states[self.layer_idx]
        if cache_params.has_previous_state and seq_len == 1 and (cache_params.conv_states[self.layer_idx].shape[0] == batch_size):
            conv_state = cache_params.conv_states[self.layer_idx]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states[:, :, 0]
            cache_params.conv_states[self.layer_idx] = conv_state
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
        else:
            conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
    else:
        ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype)
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    time_step = self.dt_layernorm(time_step)
    B = self.b_layernorm(B)
    C = self.c_layernorm(C)
    discrete_time_step = self.dt_proj(time_step)
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)
    A = -torch.exp(self.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = scan_output + hidden_states * self.D[None, :, None]
    scan_output = scan_output * self.act(gate)
    if use_cache:
        cache_params.ssm_states[self.layer_idx] = ssm_state
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))
    return contextualized_states