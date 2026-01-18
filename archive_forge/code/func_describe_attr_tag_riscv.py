from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_attr_tag_riscv(tag, val, extra):
    idx = ENUM_ATTR_TAG_RISCV[tag] - 1
    d_entry = _DESCR_ATTR_VAL_RISCV[idx]
    if d_entry is None:
        s = _DESCR_ATTR_TAG_RISCV[tag]
        s += '"%s"' % val if val else ''
        return s
    else:
        return _DESCR_ATTR_TAG_RISCV[tag] + d_entry[val]