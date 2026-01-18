import pickle
import os
import warnings
import io
from pathlib import Path
from .compressor import lz4, LZ4_NOT_INSTALLED_ERROR
from .compressor import _COMPRESSORS, register_compressor, BinaryZlibFile
from .compressor import (ZlibCompressorWrapper, GzipCompressorWrapper,
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_utils import _ensure_native_byte_order
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
from .numpy_pickle_compat import ZNDArrayWrapper  # noqa
from .backports import make_memmap
def safe_get_numpy_array_alignment_bytes(self):
    return getattr(self, 'numpy_array_alignment_bytes', None)