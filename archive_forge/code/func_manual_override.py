import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def manual_override(self):
    if not torch.cuda.is_available():
        return
    override_value = os.environ.get('BNB_CUDA_VERSION')
    if not override_value:
        return
    binary_name_stem, _, binary_name_ext = self.binary_name.rpartition('.')
    binary_name_stem = binary_name_stem.rstrip('0123456789')
    self.binary_name = f'{binary_name_stem}{override_value}.{binary_name_ext}'
    warn(f'\n\n{'=' * 80}\nWARNING: Manual override via BNB_CUDA_VERSION env variable detected!\nBNB_CUDA_VERSION=XXX can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\nIf this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\nIf you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\nFor example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\nLoading: {self.binary_name}\n{'=' * 80}\n\n')