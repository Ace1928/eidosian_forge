import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
def c_config_spaces(x):
    """constructor for ConfigSpaces"""
    ret = CConfigSpaces()
    ret.spaces_size = len(x.spaces)
    ret.spaces_key = c_str_array(x.spaces.keys())
    ret.spaces_val = c_array(CConfigSpace, [c_config_space(c) for c in x.spaces.values()])
    return ret