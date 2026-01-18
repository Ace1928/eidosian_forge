import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_init_from_file(self, model_file_path: str, thread_count: int) -> RWKVContext:
    """
        Loads the model from a file and prepares it for inference.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        model_file_path : str
            Path to model file in ggml format.
        thread_count : int
            Count of threads to use, must be positive.
        """
    ptr = self.library.rwkv_init_from_file(model_file_path.encode('utf-8'), ctypes.c_uint32(thread_count))
    if ptr is None:
        raise ValueError('rwkv_init_from_file failed, check stderr')
    return RWKVContext(ptr)