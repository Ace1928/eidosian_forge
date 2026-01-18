import contextlib
from typing import Union
import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend
def can_use_efficient_attention(params: SDPAParams, debug: bool=False) -> bool:
    """Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_mem_efficient_attention(params, debug)