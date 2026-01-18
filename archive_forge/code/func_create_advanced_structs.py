from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def create_advanced_structs(self, e_type=None, e_machine=None, e_ident_osabi=None):
    """ Create all ELF structs except the ehdr. They may possibly depend
            on provided e_type and/or e_machine parsed from ehdr.
        """
    self.e_type = e_type
    self.e_machine = e_machine
    self.e_ident_osabi = e_ident_osabi
    self._create_phdr()
    self._create_shdr()
    self._create_chdr()
    self._create_sym()
    self._create_rel()
    self._create_dyn()
    self._create_sunw_syminfo()
    self._create_gnu_verneed()
    self._create_gnu_verdef()
    self._create_gnu_versym()
    self._create_gnu_abi()
    self._create_gnu_property()
    self._create_note(e_type)
    self._create_stabs()
    self._create_attributes_subsection()
    self._create_arm_attributes()
    self._create_riscv_attributes()
    self._create_elf_hash()
    self._create_gnu_hash()