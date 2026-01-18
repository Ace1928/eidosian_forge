from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gemma.configuration_gemma import GemmaConfig
@torch.jit.script
def lm_head_with_loss(embed_weights: torch.Tensor, hidden_states: torch.Tensor, nz_shifted_label_ids: torch.Tensor, nz_shifted_loss_weights: torch.Tensor):
    logits = F.linear(hidden_states, embed_weights)
    loss = (nz_shifted_loss_weights * torch.nn.functional.cross_entropy(logits, nz_shifted_label_ids, reduction='none')).sum()
    token_accuracy = (nz_shifted_loss_weights * (torch.argmax(logits.detach(), dim=-1) == nz_shifted_label_ids)).sum()
    return (loss, token_accuracy)