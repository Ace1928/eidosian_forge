from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_ei_data(x):
    return _DESCR_EI_DATA.get(x, _unknown)