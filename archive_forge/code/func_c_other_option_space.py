import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
def c_other_option_space(x):
    """constructor for OtherOptionSpace"""
    ret = COtherOptionSpace()
    ret.entities = c_array(COtherOptionEntity, [c_other_option_entity(e) for e in x.entities])
    ret.entities_size = len(x.entities)
    return ret