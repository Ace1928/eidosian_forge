from logging import getLogger
from typing import Optional, Union
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D
from .constants import BLOCK_PATTERNS, SEQLEN_KEYS_TRANFORMERS
def get_block_name_with_pattern(model: nn.Module):
    """
    Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any((pattern_candidate in name for name in modules_names)):
            return pattern_candidate
    raise ValueError('Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`')