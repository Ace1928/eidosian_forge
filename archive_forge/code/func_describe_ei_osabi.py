from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_ei_osabi(x):
    return _DESCR_EI_OSABI.get(x, _unknown)