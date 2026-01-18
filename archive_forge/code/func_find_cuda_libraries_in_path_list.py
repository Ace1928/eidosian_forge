import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator
import torch
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import NONPYTORCH_DOC_URL
from bitsandbytes.cuda_specs import CUDASpecs
from bitsandbytes.diagnostics.utils import print_dedented
def find_cuda_libraries_in_path_list(paths_list_candidate: str) -> Iterable[Path]:
    for dir_string in paths_list_candidate.split(os.pathsep):
        if not dir_string:
            continue
        if os.sep not in dir_string:
            continue
        try:
            dir = Path(dir_string)
            try:
                if not dir.exists():
                    logger.warning(f'The directory listed in your path is found to be non-existent: {dir}')
                    continue
            except OSError:
                pass
            for lib_pattern in CUDA_RUNTIME_LIB_PATTERNS:
                for pth in dir.glob(lib_pattern):
                    if pth.is_file():
                        yield pth
        except (OSError, PermissionError):
            pass