from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def create_basic_structs(self):
    """ Create word-size related structs and ehdr struct needed for
            initial determining of ELF type.
        """
    if self.little_endian:
        self.Elf_byte = ULInt8
        self.Elf_half = ULInt16
        self.Elf_word = ULInt32
        self.Elf_word64 = ULInt64
        self.Elf_addr = ULInt32 if self.elfclass == 32 else ULInt64
        self.Elf_offset = self.Elf_addr
        self.Elf_sword = SLInt32
        self.Elf_xword = ULInt32 if self.elfclass == 32 else ULInt64
        self.Elf_sxword = SLInt32 if self.elfclass == 32 else SLInt64
    else:
        self.Elf_byte = UBInt8
        self.Elf_half = UBInt16
        self.Elf_word = UBInt32
        self.Elf_word64 = UBInt64
        self.Elf_addr = UBInt32 if self.elfclass == 32 else UBInt64
        self.Elf_offset = self.Elf_addr
        self.Elf_sword = SBInt32
        self.Elf_xword = UBInt32 if self.elfclass == 32 else UBInt64
        self.Elf_sxword = SBInt32 if self.elfclass == 32 else SBInt64
    self._create_ehdr()
    self._create_leb128()
    self._create_ntbs()