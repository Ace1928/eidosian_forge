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
def display_symbol_tables(self):
    """ Display the symbol tables contained in the file
        """
    self._init_versioninfo()
    symbol_tables = [(idx, s) for idx, s in enumerate(self.elffile.iter_sections()) if isinstance(s, SymbolTableSection)]
    if not symbol_tables and self.elffile.num_sections() == 0:
        self._emitline('')
        self._emitline('Dynamic symbol information is not available for displaying symbols.')
    for section_index, section in symbol_tables:
        if not isinstance(section, SymbolTableSection):
            continue
        if section['sh_entsize'] == 0:
            self._emitline("\nSymbol table '%s' has a sh_entsize of zero!" % section.name)
            continue
        self._emitline("\nSymbol table '%s' contains %d %s:" % (section.name, section.num_symbols(), 'entry' if section.num_symbols() == 1 else 'entries'))
        if self.elffile.elfclass == 32:
            self._emitline('   Num:    Value  Size Type    Bind   Vis      Ndx Name')
        else:
            self._emitline('   Num:    Value          Size Type    Bind   Vis      Ndx Name')
        for nsym, symbol in enumerate(section.iter_symbols()):
            version_info = ''
            if section['sh_type'] == 'SHT_DYNSYM' and self._versioninfo['type'] == 'GNU':
                version = self._symbol_version(nsym)
                if version['name'] != symbol.name and version['index'] not in ('VER_NDX_LOCAL', 'VER_NDX_GLOBAL'):
                    if version['filename']:
                        version_info = '@%(name)s (%(index)i)' % version
                    elif version['hidden']:
                        version_info = '@%(name)s' % version
                    else:
                        version_info = '@@%(name)s' % version
            symbol_name = symbol.name
            if symbol['st_info']['type'] == 'STT_SECTION' and symbol['st_shndx'] < self.elffile.num_sections() and (symbol['st_name'] == 0):
                symbol_name = self.elffile.get_section(symbol['st_shndx']).name
            self._emitline('%6d: %s %s %-7s %-6s %-7s %4s %.25s%s' % (nsym, self._format_hex(symbol['st_value'], fullhex=True, lead0x=False), '%5d' % symbol['st_size'] if symbol['st_size'] < 100000 else hex(symbol['st_size']), describe_symbol_type(symbol['st_info']['type']), describe_symbol_bind(symbol['st_info']['bind']), describe_symbol_other(symbol['st_other']), describe_symbol_shndx(self._get_symbol_shndx(symbol, nsym, section_index)), _format_symbol_name(symbol_name), version_info))