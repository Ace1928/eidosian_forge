import pickle
import os
import zlib
import inspect
from io import BytesIO
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler
from .numpy_pickle_utils import _ensure_native_byte_order
def read_zfile(file_handle):
    """Read the z-file and return the content as a string.

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guaranteed. Do not
    use for external purposes.
    """
    file_handle.seek(0)
    header_length = len(_ZFILE_PREFIX) + _MAX_LEN
    length = file_handle.read(header_length)
    length = length[len(_ZFILE_PREFIX):]
    length = int(length, 16)
    next_byte = file_handle.read(1)
    if next_byte != b' ':
        file_handle.seek(header_length)
    data = zlib.decompress(file_handle.read(), 15, length)
    assert len(data) == length, 'Incorrect data length while decompressing %s.The file could be corrupted.' % file_handle
    return data