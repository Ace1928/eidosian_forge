import contextlib
from typing import Union
import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend
@contextlib.contextmanager
def sdp_kernel(enable_flash: bool=True, enable_math: bool=True, enable_mem_efficient: bool=True):
    """
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    previous_flash: bool = flash_sdp_enabled()
    previous_mem_efficient: bool = mem_efficient_sdp_enabled()
    previous_math: bool = math_sdp_enabled()
    try:
        enable_flash_sdp(enable_flash)
        enable_mem_efficient_sdp(enable_mem_efficient)
        enable_math_sdp(enable_math)
        yield {}
    finally:
        enable_flash_sdp(previous_flash)
        enable_mem_efficient_sdp(previous_mem_efficient)
        enable_math_sdp(previous_math)