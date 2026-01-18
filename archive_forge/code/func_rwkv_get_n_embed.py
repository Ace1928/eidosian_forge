import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_get_n_embed(self, ctx: RWKVContext) -> int:
    """
        Returns the number of elements in the given model's embedding.
        Useful for reading individual fields of a model's hidden state.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """
    return self.library.rwkv_get_n_embed(ctx.ptr)