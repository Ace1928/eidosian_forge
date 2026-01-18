from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def num_relocations(self):
    """ Number of relocations in the section
        """
    if self._cached_relocations is None:
        self._cached_relocations = list(self.iter_relocations())
    return len(self._cached_relocations)