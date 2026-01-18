import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def remove_non_existent_dirs(candidate_paths: Set[Path]) -> Set[Path]:
    existent_directories: Set[Path] = set()
    for path in candidate_paths:
        try:
            if path.exists():
                existent_directories.add(path)
        except PermissionError:
            pass
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise exc
    non_existent_directories: Set[Path] = candidate_paths - existent_directories
    if non_existent_directories:
        CUDASetup.get_instance().add_log_entry(f'The following directories listed in your path were found to be non-existent: {non_existent_directories}', is_warning=False)
    return existent_directories