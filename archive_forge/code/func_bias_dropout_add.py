from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig
def bias_dropout_add(x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    """add bias to x, apply dropout and residual connection

    Args:
        x (Tensor): main path of output
        bias (Tensor): None or attn_bias of the last attention layer
        residual (Optional[Tensor]): residual value
        prob (float): dropout probability
        training (bool): whether in training mode or not

    Returns:
        Tensor: dropout(x + bias) + residual
    """
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out