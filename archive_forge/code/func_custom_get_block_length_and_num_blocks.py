from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging
def custom_get_block_length_and_num_blocks(seq_length, window_size):
    """
    Custom implementation for GPTNeoAttentionMixin._get_block_length_and_num_blocks to enable the export to ONNX as
    original implementation uses Python variables and control flow.
    """
    import torch
    candidates = torch.arange(1, window_size)
    remainders = torch.remainder(seq_length, candidates)
    divisor_indices = remainders == 0
    divisors = candidates[divisor_indices]
    largest_divisor = torch.max(divisors)
    return (largest_divisor, torch.div(seq_length, largest_divisor, rounding_mode='floor'))