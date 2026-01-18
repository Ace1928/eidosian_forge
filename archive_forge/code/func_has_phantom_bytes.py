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
def has_phantom_bytes(self):
    """The XC16 compiler for the PIC microcontrollers emits DWARF where all odd bytes in all DWARF sections
           are to be discarded ("phantom").

            We don't know where does the phantom byte discarding fit into the usual chain of section content transforms.
            There are no XC16/PIC binaries in the corpus with relocations against DWARF, and the DWARF section compression
            seems to be unsupported by XC16.
        """
    return self['e_machine'] == 'EM_DSPIC30F' and self['e_flags'] & 2147483648 == 0