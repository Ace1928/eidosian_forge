import logging
import os
import sys
from torchaudio._internal.module_utils import fail_with_message, is_module_available, no_op
from .utils import _check_cuda_version, _init_dll_path, _init_sox, _LazyImporter, _load_lib
def lazy_import_sox_ext():
    """Load SoX integration based on availability in lazy manner"""
    global _SOX_EXT
    if _SOX_EXT is None:
        _SOX_EXT = _LazyImporter('_torchaudio_sox', _init_sox)
    return _SOX_EXT