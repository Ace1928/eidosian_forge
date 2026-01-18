import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def show_flag_values(value):
    return list(_iter_bits_lsb(value))