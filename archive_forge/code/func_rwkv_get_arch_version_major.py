import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_get_arch_version_major(self, ctx: RWKVContext) -> int:
    """
        Returns the major version used by the given model.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """
    return self.library.rwkv_get_arch_version_major(ctx.ptr)