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
def get_supplementary_dwarfinfo(self, dwarfinfo):
    """
        Read supplementary dwarfinfo, from either the standared .debug_sup
        section or the GNU proprietary .gnu_debugaltlink.
        """
    supfilepath = dwarfinfo.parse_debugsupinfo()
    if supfilepath is not None and self.stream_loader is not None:
        stream = self.stream_loader(supfilepath)
        supelffile = ELFFile(stream)
        dwarf_info = supelffile.get_dwarf_info()
        stream.close()
        return dwarf_info
    return None