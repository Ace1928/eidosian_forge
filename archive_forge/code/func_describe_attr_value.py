from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def describe_attr_value(attr, die, section_offset):
    """ Given an attribute attr, return the textual representation of its
        value, suitable for tools like readelf.

        To cover all cases, this function needs some extra arguments:

        die: the DIE this attribute was extracted from
        section_offset: offset in the stream of the section the DIE belongs to
    """
    descr_func = _ATTR_DESCRIPTION_MAP[attr.form]
    val_description = descr_func(attr, die, section_offset)
    extra_info_func = _EXTRA_INFO_DESCRIPTION_MAP[attr.name]
    extra_info = extra_info_func(attr, die, section_offset)
    return str(val_description) + '\t' + extra_info