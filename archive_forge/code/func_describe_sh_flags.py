from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_sh_flags(x):
    s = ''
    for flag in (SH_FLAGS.SHF_WRITE, SH_FLAGS.SHF_ALLOC, SH_FLAGS.SHF_EXECINSTR, SH_FLAGS.SHF_MERGE, SH_FLAGS.SHF_STRINGS, SH_FLAGS.SHF_INFO_LINK, SH_FLAGS.SHF_LINK_ORDER, SH_FLAGS.SHF_OS_NONCONFORMING, SH_FLAGS.SHF_GROUP, SH_FLAGS.SHF_TLS, SH_FLAGS.SHF_MASKOS, SH_FLAGS.SHF_EXCLUDE):
        s += _DESCR_SH_FLAGS[flag] if x & flag else ''
    if not x & SH_FLAGS.SHF_EXCLUDE:
        if x & SH_FLAGS.SHF_MASKPROC:
            s += 'p'
    return s