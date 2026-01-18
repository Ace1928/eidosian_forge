from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_dyn_tag(x):
    return _DESCR_D_TAG.get(x, _unknown)