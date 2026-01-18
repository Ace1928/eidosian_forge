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
def range_lists(self):
    """ Get a RangeLists object representing the .debug_ranges/.debug_rnglists section of
            the DWARF data, or None if this section doesn't exist.

            If both sections exist, it returns a RangeListsPair.
        """
    if self.debug_rnglists_sec and self.debug_ranges_sec is None:
        return RangeLists(self.debug_rnglists_sec.stream, self.structs, 5, self)
    elif self.debug_ranges_sec and self.debug_rnglists_sec is None:
        return RangeLists(self.debug_ranges_sec.stream, self.structs, 4, self)
    elif self.debug_ranges_sec and self.debug_rnglists_sec:
        return RangeListsPair(self.debug_ranges_sec.stream, self.debug_rnglists_sec.stream, self.structs, self)
    else:
        return None