import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array
from ..base import NDArrayHandle, CachedOpHandle
from ..base import check_call
from .. import _global_var
def callback_handle(name, opr_name, array, _):
    """ ctypes function """
    callback(name, opr_name, array)