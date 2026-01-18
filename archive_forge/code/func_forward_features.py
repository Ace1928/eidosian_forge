import math
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.helpers import named_apply
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth
from flash_attn.layers.patch_embed import PatchEmbed
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
def forward_features(self, x, all_tokens=True):
    """
        If all_tokens==False and self.global_pool == 'token', we only return the features for the
        cls token.
        """
    x = self.patch_embed(x)
    hidden_states = self._pos_embed(x)
    residual = None
    if self.global_pool != 'token' or all_tokens:
        for block in self.blocks:
            hidden_states, residual = block(hidden_states, residual)
    else:
        for block in self.blocks[:-1]:
            hidden_states, residual = block(hidden_states, residual)
        hidden_states, residual = self.blocks[-1](hidden_states, residual, mixer_subset=slice(0, 1))
    if not self.fused_dropout_add_ln:
        residual = self.drop_path(self.dropout(hidden_states)) + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
    else:
        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(torch.ones(hidden_states.shape[:-1], device=hidden_states.device, dtype=hidden_states.dtype))
        hidden_states = layer_norm_fn(hidden_states, self.norm.weight, self.norm.bias, residual=residual, eps=self.norm.eps, dropout_p=self.dropout.p if self.training else 0.0, rowscale=rowscale, prenorm=False)
    return hidden_states