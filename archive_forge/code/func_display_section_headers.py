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
def display_section_headers(self, show_heading=True):
    """ Display the ELF section headers
        """
    elfheader = self.elffile.header
    if show_heading:
        self._emitline('There are %s section headers, starting at offset %s' % (elfheader['e_shnum'], self._format_hex(elfheader['e_shoff'])))
    if self.elffile.num_sections() == 0:
        self._emitline('There are no sections in this file.')
        return
    self._emitline('\nSection Header%s:' % ('s' if self.elffile.num_sections() > 1 else ''))
    if self.elffile.elfclass == 32:
        self._emitline('  [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al')
    else:
        self._emitline('  [Nr] Name              Type             Address           Offset')
        self._emitline('       Size              EntSize          Flags  Link  Info  Align')
    for nsec, section in enumerate(self.elffile.iter_sections()):
        self._emit('  [%2u] %-17.17s %-15.15s ' % (nsec, section.name, describe_sh_type(section['sh_type'])))
        if self.elffile.elfclass == 32:
            self._emitline('%s %s %s %s %3s %2s %3s %2s' % (self._format_hex(section['sh_addr'], fieldsize=8, lead0x=False), self._format_hex(section['sh_offset'], fieldsize=6, lead0x=False), self._format_hex(section['sh_size'], fieldsize=6, lead0x=False), self._format_hex(section['sh_entsize'], fieldsize=2, lead0x=False), describe_sh_flags(section['sh_flags']), section['sh_link'], section['sh_info'], section['sh_addralign']))
        else:
            self._emitline(' %s  %s' % (self._format_hex(section['sh_addr'], fullhex=True, lead0x=False), self._format_hex(section['sh_offset'], fieldsize=16 if section['sh_offset'] > 4294967295 else 8, lead0x=False)))
            self._emitline('       %s  %s %3s      %2s   %3s     %s' % (self._format_hex(section['sh_size'], fullhex=True, lead0x=False), self._format_hex(section['sh_entsize'], fullhex=True, lead0x=False), describe_sh_flags(section['sh_flags']), section['sh_link'], section['sh_info'], section['sh_addralign']))
    self._emitline('Key to Flags:')
    self._emitline('  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),')
    self._emitline('  L (link order), O (extra OS processing required), G (group), T (TLS),')
    self._emitline('  C (compressed), x (unknown), o (OS specific), E (exclude),')
    self._emit('  ')
    if self.elffile['e_machine'] == 'EM_ARM':
        self._emit('y (purecode), ')
    self._emitline('p (processor specific)')