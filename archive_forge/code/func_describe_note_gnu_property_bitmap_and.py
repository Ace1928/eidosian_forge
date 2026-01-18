from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_note_gnu_property_bitmap_and(values, prefix, value):
    descs = []
    for mask, desc in values:
        if value & mask:
            descs.append(desc)
    return '%s: %s' % (prefix, ', '.join(descs))