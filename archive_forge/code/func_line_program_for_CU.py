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
def line_program_for_CU(self, CU):
    """ Given a CU object, fetch the line program it points to from the
            .debug_line section.
            If the CU doesn't point to a line program, return None.

            Note about directory and file names. They are returned as two collections
            in the lineprogram object's header - include_directory and file_entry.

            In DWARFv5, they have introduced a different, extensible format for those
            collections. So in a lineprogram v5+, there are two more collections in
            the header - directories and file_names. Those might contain extra DWARFv5
            information that is not exposed in include_directory and file_entry.
        """
    top_DIE = CU.get_top_DIE()
    if 'DW_AT_stmt_list' in top_DIE.attributes:
        return self._parse_line_program_at_offset(top_DIE.attributes['DW_AT_stmt_list'].value, CU.structs)
    else:
        return None