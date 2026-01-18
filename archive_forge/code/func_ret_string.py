import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
def ret_string(ptr):
    """To wrap string return-value from C-API.
    """
    if ptr is not None:
        return str(OutputString.from_return(ptr))