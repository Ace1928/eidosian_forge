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
def get_CU_containing(self, refaddr):
    """ Find the CU that includes the given reference address in the
            .debug_info section.

            refaddr:
                Either a refaddr of a DIE (possibly from a DW_FORM_ref_addr
                attribute) or the section offset of a CU (possibly from an
                aranges table).

           This function will parse and cache CUs until the search criteria
           is met, starting from the closest known offset lessthan or equal
           to the given address.
        """
    dwarf_assert(self.has_debug_info, 'CU lookup but no debug info section')
    dwarf_assert(0 <= refaddr < self.debug_info_sec.size, 'refaddr %s beyond .debug_info size' % refaddr)
    i = bisect_right(self._cu_offsets_map, refaddr)
    start = self._cu_offsets_map[i - 1] if i > 0 else 0
    for cu in self._parse_CUs_iter(start):
        if cu.cu_offset <= refaddr < cu.cu_offset + cu.size:
            return cu
    raise ValueError('CU for reference address %s not found' % refaddr)