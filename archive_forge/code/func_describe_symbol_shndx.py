from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_symbol_shndx(x):
    return _DESCR_ST_SHNDX.get(x, '%3s' % x)