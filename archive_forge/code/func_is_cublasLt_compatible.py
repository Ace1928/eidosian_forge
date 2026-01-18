import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def is_cublasLt_compatible(cc):
    has_cublaslt = False
    if cc is not None:
        cc_major, cc_minor = cc.split('.')
        if int(cc_major) < 7 or (int(cc_major) == 7 and int(cc_minor) < 5):
            CUDASetup.get_instance().add_log_entry('WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!                     If you run into issues with 8-bit matmul, you can try 4-bit quantization: https://huggingface.co/blog/4bit-transformers-bitsandbytes', is_warning=True)
        else:
            has_cublaslt = True
    return has_cublaslt