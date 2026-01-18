from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def describe_form_class(form):
    """For a given form name, determine its value class.

    For example, given 'DW_FORM_data1' returns 'constant'.

    For some forms, like DW_FORM_indirect and DW_FORM_sec_offset, the class is
    not hard-coded and extra information is required. For these, None is
    returned.
    """
    return _FORM_CLASS[form]