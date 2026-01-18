import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_get_n_vocab(self, ctx: RWKVContext) -> int:
    """
        Returns the number of tokens in the given model's vocabulary.
        Useful for telling 20B_tokenizer models (n_vocab = 50277) apart from World models (n_vocab = 65536).

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """
    return self.library.rwkv_get_n_vocab(ctx.ptr)