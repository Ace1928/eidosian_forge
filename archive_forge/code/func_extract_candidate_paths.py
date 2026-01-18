import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def extract_candidate_paths(paths_list_candidate: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths_list_candidate.split(os.pathsep) if ld_path}