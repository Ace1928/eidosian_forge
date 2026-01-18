import argparse
import os, sys
import re
import string
import traceback
import itertools
from elftools import __version__
from elftools.common.exceptions import ELFError
from elftools.common.utils import bytes2str, iterbytes
from elftools.elf.elffile import ELFFile
from elftools.elf.dynamic import DynamicSection, DynamicSegment
from elftools.elf.enums import ENUM_D_TAG
from elftools.elf.segments import InterpSegment
from elftools.elf.sections import (
from elftools.elf.gnuversions import (
from elftools.elf.relocation import RelocationSection
from elftools.elf.descriptions import (
from elftools.elf.constants import E_FLAGS
from elftools.elf.constants import E_FLAGS_MASKS
from elftools.elf.constants import SH_FLAGS
from elftools.elf.constants import SHN_INDICES
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.descriptions import (
from elftools.dwarf.constants import (
from elftools.dwarf.locationlists import LocationParser, LocationEntry, LocationViewPair, BaseAddressEntry as LocBaseAddressEntry, LocationListsPair
from elftools.dwarf.ranges import RangeEntry, BaseAddressEntry as RangeBaseAddressEntry, RangeListsPair
from elftools.dwarf.callframe import CIE, FDE, ZERO
from elftools.ehabi.ehabiinfo import CorruptEHABIEntry, CannotUnwindEHABIEntry, GenericEHABIEntry
from elftools.dwarf.enums import ENUM_DW_UT
def display_debug_dump(self, dump_what):
    """ Dump a DWARF section
        """
    self._init_dwarfinfo()
    if self._dwarfinfo is None:
        return
    set_global_machine_arch(self.elffile.get_machine_arch())
    if dump_what == 'info':
        self._dump_debug_info()
    elif dump_what == 'decodedline':
        self._dump_debug_line_programs()
    elif dump_what == 'frames':
        self._dump_debug_frames()
    elif dump_what == 'frames-interp':
        self._dump_debug_frames_interp()
    elif dump_what == 'aranges':
        self._dump_debug_aranges()
    elif dump_what in {'pubtypes', 'pubnames'}:
        self._dump_debug_namelut(dump_what)
    elif dump_what == 'loc':
        self._dump_debug_locations()
    elif dump_what == 'Ranges':
        self._dump_debug_ranges()
    else:
        self._emitline('debug dump not yet supported for "%s"' % dump_what)