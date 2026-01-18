import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def find_cuda_lib_in(paths_list_candidate: str) -> Set[Path]:
    return get_cuda_runtime_lib_paths(resolve_paths_list(paths_list_candidate))