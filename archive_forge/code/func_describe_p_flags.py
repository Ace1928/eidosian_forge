from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_p_flags(x):
    s = ''
    for flag in (P_FLAGS.PF_R, P_FLAGS.PF_W, P_FLAGS.PF_X):
        s += _DESCR_P_FLAGS[flag] if x & flag else ' '
    return s