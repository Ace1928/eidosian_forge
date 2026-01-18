import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
def c_other_option_entity(x):
    """constructor for OtherOptionEntity"""
    ret = COtherOptionEntity()
    ret.val = x.val
    return ret