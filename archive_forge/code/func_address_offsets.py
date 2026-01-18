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
def address_offsets(self, start, size=1):
    """ Yield a file offset for each ELF segment containing a memory region.

            A memory region is defined by the range [start...start+size). The
            offset of the region is yielded.
        """
    end = start + size
    for seg in self.iter_segments(type='PT_LOAD'):
        if start >= seg['p_vaddr'] and end <= seg['p_vaddr'] + seg['p_filesz']:
            yield (start - seg['p_vaddr'] + seg['p_offset'])