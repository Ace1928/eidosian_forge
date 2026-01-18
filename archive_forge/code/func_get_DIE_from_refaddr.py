import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def get_DIE_from_refaddr(self, refaddr, cu=None):
    """ Given a .debug_info section offset of a DIE, return the DIE.

            refaddr:
                The refaddr may come from a DW_FORM_ref_addr attribute.

            cu:
                The compile unit object, if known.  If None a search
                from the closest offset less than refaddr will be performed.
        """
    if cu is None:
        cu = self.get_CU_containing(refaddr)
    return cu.get_DIE_from_refaddr(refaddr)