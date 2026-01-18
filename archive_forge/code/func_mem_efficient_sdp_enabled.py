import contextlib
from typing import Union
import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend
def mem_efficient_sdp_enabled():
    """
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    return torch._C._get_mem_efficient_sdp_enabled()