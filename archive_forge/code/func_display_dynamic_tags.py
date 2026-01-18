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
def display_dynamic_tags(self):
    """ Display the dynamic tags contained in the file
        """
    has_dynamic_sections = False
    for section in self.elffile.iter_sections():
        if not isinstance(section, DynamicSection):
            continue
        has_dynamic_sections = True
        self._emitline('\nDynamic section at offset %s contains %d %s:' % (self._format_hex(section['sh_offset']), section.num_tags(), 'entry' if section.num_tags() == 1 else 'entries'))
        self._emitline('  Tag        Type                         Name/Value')
        padding = 20 + (8 if self.elffile.elfclass == 32 else 0)
        for tag in section.iter_tags():
            if tag.entry.d_tag == 'DT_NEEDED':
                parsed = 'Shared library: [%s]' % tag.needed
            elif tag.entry.d_tag == 'DT_RPATH':
                parsed = 'Library rpath: [%s]' % tag.rpath
            elif tag.entry.d_tag == 'DT_RUNPATH':
                parsed = 'Library runpath: [%s]' % tag.runpath
            elif tag.entry.d_tag == 'DT_SONAME':
                parsed = 'Library soname: [%s]' % tag.soname
            elif tag.entry.d_tag.endswith(('SZ', 'ENT')):
                parsed = '%i (bytes)' % tag['d_val']
            elif tag.entry.d_tag == 'DT_FLAGS':
                parsed = describe_dt_flags(tag.entry.d_val)
            elif tag.entry.d_tag == 'DT_FLAGS_1':
                parsed = 'Flags: %s' % describe_dt_flags_1(tag.entry.d_val)
            elif tag.entry.d_tag.endswith(('NUM', 'COUNT')):
                parsed = '%i' % tag['d_val']
            elif tag.entry.d_tag == 'DT_PLTREL':
                s = describe_dyn_tag(tag.entry.d_val)
                if s.startswith('DT_'):
                    s = s[3:]
                parsed = '%s' % s
            elif tag.entry.d_tag == 'DT_MIPS_FLAGS':
                parsed = describe_rh_flags(tag.entry.d_val)
            elif tag.entry.d_tag in ('DT_MIPS_SYMTABNO', 'DT_MIPS_LOCAL_GOTNO'):
                parsed = str(tag.entry.d_val)
            elif tag.entry.d_tag == 'DT_AARCH64_BTI_PLT':
                parsed = ''
            else:
                parsed = '%#x' % tag['d_val']
            self._emitline(' %s %-*s %s' % (self._format_hex(ENUM_D_TAG.get(tag.entry.d_tag, tag.entry.d_tag), fullhex=True, lead0x=True), padding, '(%s)' % (tag.entry.d_tag[3:],), parsed))
    if not has_dynamic_sections:
        self._emitline('\nThere is no dynamic section in this file.')