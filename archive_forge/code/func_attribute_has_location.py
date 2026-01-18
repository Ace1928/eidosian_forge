import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
@staticmethod
def attribute_has_location(attr, dwarf_version):
    """ Checks if a DIE attribute contains location information.
        """
    return LocationParser._attribute_is_loclistptr_class(attr) and (LocationParser._attribute_has_loc_expr(attr, dwarf_version) or LocationParser._attribute_has_loc_list(attr, dwarf_version))