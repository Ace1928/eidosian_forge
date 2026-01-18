import io
from io import BytesIO
import os
import struct
import zlib
from ..common.exceptions import ELFError, ELFParseError
from ..common.utils import struct_parse, elf_assert
from .structs import ELFStructs
from .sections import (
from .dynamic import DynamicSection, DynamicSegment
from .relocation import (RelocationSection, RelocationHandler,
from .gnuversions import (
from .segments import Segment, InterpSegment, NoteSegment
from ..dwarf.dwarfinfo import DWARFInfo, DebugSectionDescriptor, DwarfConfig
from ..ehabi.ehabiinfo import EHABIInfo
from .hash import ELFHashSection, GNUHashSection
from .constants import SHN_INDICES
def get_section_by_name(self, name):
    """ Get a section from the file, by name. Return None if no such
            section exists.
        """
    if self._section_name_map is None:
        self._make_section_name_map()
    secnum = self._section_name_map.get(name, None)
    return None if secnum is None else self.get_section(secnum)