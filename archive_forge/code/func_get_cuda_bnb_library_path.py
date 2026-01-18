import ctypes as ct
import logging
import os
from pathlib import Path
import torch
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs
def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the disk path to the CUDA BNB native library specified by the
    given CUDA specs, taking into account the `BNB_CUDA_VERSION` override environment variable.

    The library is not guaranteed to exist at the returned path.
    """
    library_name = f'libbitsandbytes_cuda{cuda_specs.cuda_version_string}'
    if not cuda_specs.has_cublaslt:
        library_name += '_nocublaslt'
    library_name = f'{library_name}{DYNAMIC_LIBRARY_SUFFIX}'
    override_value = os.environ.get('BNB_CUDA_VERSION')
    if override_value:
        library_name_stem, _, library_name_ext = library_name.rpartition('.')
        library_name_stem = library_name_stem.rstrip('0123456789')
        library_name = f'{library_name_stem}{override_value}.{library_name_ext}'
        logger.warning(f'WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\nThis can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\nIf this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\nIf you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\nFor example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n')
    return PACKAGE_DIR / library_name