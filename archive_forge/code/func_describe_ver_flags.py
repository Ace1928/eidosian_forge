from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_ver_flags(x):
    return ' | '.join((_DESCR_VER_FLAGS[flag] for flag in (VER_FLAGS.VER_FLG_WEAK, VER_FLAGS.VER_FLG_BASE, VER_FLAGS.VER_FLG_INFO) if x & flag))