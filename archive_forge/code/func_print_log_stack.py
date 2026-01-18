import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def print_log_stack(self):
    for msg, is_warning in self.cuda_setup_log:
        if is_warning:
            warn(msg)
        else:
            print(msg)