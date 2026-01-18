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
def location_lists(self):
    """ Get a LocationLists object representing the .debug_loc/debug_loclists section of
            the DWARF data, or None if this section doesn't exist.

            If both sections exist, it returns a LocationListsPair.
        """
    if self.debug_loclists_sec and self.debug_loc_sec is None:
        return LocationLists(self.debug_loclists_sec.stream, self.structs, 5, self)
    elif self.debug_loc_sec and self.debug_loclists_sec is None:
        return LocationLists(self.debug_loc_sec.stream, self.structs, 4, self)
    elif self.debug_loc_sec and self.debug_loclists_sec:
        return LocationListsPair(self.debug_loc_sec.stream, self.debug_loclists_sec.stream, self.structs, self)
    else:
        return None