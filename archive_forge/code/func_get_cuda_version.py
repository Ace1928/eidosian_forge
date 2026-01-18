import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def get_cuda_version():
    major, minor = map(int, torch.version.cuda.split('.'))
    if major < 11:
        CUDASetup.get_instance().add_log_entry('CUDA SETUP: CUDA version lower than 11 are currently not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!')
    return f'{major}{minor}'