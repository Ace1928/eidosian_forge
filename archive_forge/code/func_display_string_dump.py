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
def display_string_dump(self, section_spec):
    """ Display a strings dump of a section. section_spec is either a
            section number or a name.
        """
    section = self._section_from_spec(section_spec)
    if section is None:
        sys.stderr.write("readelf.py: Warning: Section '%s' was not dumped because it does not exist!\n" % section_spec)
        return
    if section['sh_type'] == 'SHT_NOBITS':
        self._emitline("\nSection '%s' has no data to dump." % section_spec)
        return
    self._emitline("\nString dump of section '%s':" % section.name)
    found = False
    data = section.data()
    dataptr = 0
    while dataptr < len(data):
        while dataptr < len(data) and (not 32 <= data[dataptr] <= 127):
            dataptr += 1
        if dataptr >= len(data):
            break
        endptr = dataptr
        while endptr < len(data) and data[endptr] != 0:
            endptr += 1
        found = True
        self._emitline('  [%6x]  %s' % (dataptr, bytes2str(data[dataptr:endptr])))
        dataptr = endptr
    if not found:
        self._emitline('  No strings found in this section.')
    else:
        self._emitline()