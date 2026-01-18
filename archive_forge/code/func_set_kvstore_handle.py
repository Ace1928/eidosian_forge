import ctypes
import warnings
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array, py_str, KVStoreHandle
def set_kvstore_handle(handle):
    global profiler_kvstore_handle
    profiler_kvstore_handle = handle