import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
def get_relocation_tables(self):
    """ Load all available relocation tables from DYNAMIC tags.

            Returns a dictionary mapping found table types (REL, RELA,
            RELR, JMPREL) to RelocationTable objects.
        """
    result = {}
    if list(self.iter_tags('DT_REL')):
        result['REL'] = RelocationTable(self.elffile, self.get_table_offset('DT_REL')[1], next(self.iter_tags('DT_RELSZ'))['d_val'], False)
        relentsz = next(self.iter_tags('DT_RELENT'))['d_val']
        elf_assert(result['REL'].entry_size == relentsz, 'Expected DT_RELENT to be %s' % relentsz)
    if list(self.iter_tags('DT_RELA')):
        result['RELA'] = RelocationTable(self.elffile, self.get_table_offset('DT_RELA')[1], next(self.iter_tags('DT_RELASZ'))['d_val'], True)
        relentsz = next(self.iter_tags('DT_RELAENT'))['d_val']
        elf_assert(result['RELA'].entry_size == relentsz, 'Expected DT_RELAENT to be %s' % relentsz)
    if list(self.iter_tags('DT_RELR')):
        result['RELR'] = RelrRelocationTable(self.elffile, self.get_table_offset('DT_RELR')[1], next(self.iter_tags('DT_RELRSZ'))['d_val'], next(self.iter_tags('DT_RELRENT'))['d_val'])
    if list(self.iter_tags('DT_JMPREL')):
        result['JMPREL'] = RelocationTable(self.elffile, self.get_table_offset('DT_JMPREL')[1], next(self.iter_tags('DT_PLTRELSZ'))['d_val'], next(self.iter_tags('DT_PLTREL'))['d_val'] == ENUM_D_TAG['DT_RELA'])
    return result