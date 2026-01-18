from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_attr_tag_arm(tag, val, extra):
    idx = ENUM_ATTR_TAG_ARM[tag] - 1
    d_entry = _DESCR_ATTR_VAL_ARM[idx]
    if d_entry is None:
        if tag == 'TAG_COMPATIBILITY':
            return _DESCR_ATTR_TAG_ARM[tag] + 'flag = %d, vendor = %s' % (val, extra)
        elif tag == 'TAG_ALSO_COMPATIBLE_WITH':
            if val.tag == 'TAG_CPU_ARCH':
                return _DESCR_ATTR_TAG_ARM[tag] + d_entry[val]
            else:
                return _DESCR_ATTR_TAG_ARM[tag] + '??? (%d)' % val.tag
        elif tag == 'TAG_NODEFAULTS':
            return _DESCR_ATTR_TAG_ARM[tag] + 'True'
        s = _DESCR_ATTR_TAG_ARM[tag]
        s += '"%s"' % val if val else ''
        return s
    else:
        return _DESCR_ATTR_TAG_ARM[tag] + d_entry[val]