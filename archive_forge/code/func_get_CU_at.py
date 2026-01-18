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
def get_CU_at(self, offset):
    """ Given a CU header offset, return the parsed CU.

            offset:
                The offset may be from an accelerated access table such as
                the public names, public types, address range table, or
                prior use.

            This function will directly parse the CU doing no validation of
            the offset beyond checking the size of the .debug_info section.
        """
    dwarf_assert(self.has_debug_info, 'CU lookup but no debug info section')
    dwarf_assert(0 <= offset < self.debug_info_sec.size, 'offset %s beyond .debug_info size' % offset)
    return self._cached_CU_at_offset(offset)