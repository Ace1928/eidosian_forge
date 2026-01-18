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
def display_hex_dump(self, section_spec):
    """ Display a hex dump of a section. section_spec is either a section
            number or a name.
        """
    section = self._section_from_spec(section_spec)
    if section is None:
        sys.stderr.write("readelf: Warning: Section '%s' was not dumped because it does not exist!\n" % section_spec)
        return
    if section['sh_type'] == 'SHT_NOBITS':
        self._emitline("\nSection '%s' has no data to dump." % section_spec)
        return
    self._emitline("\nHex dump of section '%s':" % section.name)
    self._note_relocs_for_section(section)
    addr = section['sh_addr']
    data = section.data()
    dataptr = 0
    while dataptr < len(data):
        bytesleft = len(data) - dataptr
        linebytes = 16 if bytesleft > 16 else bytesleft
        self._emit('  %s ' % self._format_hex(addr, fieldsize=8))
        for i in range(16):
            if i < linebytes:
                self._emit('%2.2x' % data[dataptr + i])
            else:
                self._emit('  ')
            if i % 4 == 3:
                self._emit(' ')
        for i in range(linebytes):
            c = data[dataptr + i:dataptr + i + 1]
            if c[0] >= 32 and c[0] < 127:
                self._emit(bytes2str(c))
            else:
                self._emit(bytes2str(b'.'))
        self._emitline()
        addr += linebytes
        dataptr += linebytes
    self._emitline()