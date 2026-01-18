from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_symbol_local(x):
    return '[<localentry>: ' + str(1 << x) + ']'