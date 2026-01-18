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
def parse_debugsupinfo(self):
    """
        Extract a filename from either .debug_sup or .gnu_debualtlink sections.
        """
    if self.debug_sup_sec is not None:
        self.debug_sup_sec.stream.seek(0)
        suplink = self.structs.Dwarf_debugsup.parse_stream(self.debug_sup_sec.stream)
        if suplink.is_supplementary == 0:
            return suplink.sup_filename
    if self.gnu_debugaltlink_sec is not None:
        self.gnu_debugaltlink_sec.stream.seek(0)
        suplink = self.structs.Dwarf_debugaltlink.parse_stream(self.gnu_debugaltlink_sec.stream)
        return suplink.sup_filename
    return None