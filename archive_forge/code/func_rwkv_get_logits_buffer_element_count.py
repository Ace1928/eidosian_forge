import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_get_logits_buffer_element_count(self, ctx: RWKVContext) -> int:
    """
        Returns count of FP32 elements in logits buffer.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """
    return self.library.rwkv_get_logits_buffer_element_count(ctx.ptr)