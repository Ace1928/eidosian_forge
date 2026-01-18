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
def read_mmap(self, unpickler):
    """Read an array using numpy memmap."""
    current_pos = unpickler.file_handle.tell()
    offset = current_pos
    numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
    if numpy_array_alignment_bytes is not None:
        padding_byte = unpickler.file_handle.read(1)
        padding_length = int.from_bytes(padding_byte, byteorder='little')
        offset += padding_length + 1
    if unpickler.mmap_mode == 'w+':
        unpickler.mmap_mode = 'r+'
    marray = make_memmap(unpickler.filename, dtype=self.dtype, shape=self.shape, order=self.order, mode=unpickler.mmap_mode, offset=offset)
    unpickler.file_handle.seek(offset + marray.nbytes)
    if numpy_array_alignment_bytes is None and current_pos % NUMPY_ARRAY_ALIGNMENT_BYTES != 0:
        message = f'The memmapped array {marray} loaded from the file {unpickler.file_handle.name} is not byte aligned. This may cause segmentation faults if this memmapped array is used in some libraries like BLAS or PyTorch. To get rid of this warning, regenerate your pickle file with joblib >= 1.2.0. See https://github.com/joblib/joblib/issues/563 for more details'
        warnings.warn(message)
    return _ensure_native_byte_order(marray)