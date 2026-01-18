from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def apply_section_relocations(self, stream, reloc_section):
    """ Apply all relocations in reloc_section (a RelocationSection object)
            to the given stream, that contains the data of the section that is
            being relocated. The stream is modified as a result.
        """
    symtab = self.elffile.get_section(reloc_section['sh_link'])
    for reloc in reloc_section.iter_relocations():
        self._do_apply_relocation(stream, reloc, symtab)