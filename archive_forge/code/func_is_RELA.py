from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def is_RELA(self):
    """ Is this a RELA relocation section? If not, it's REL.
        """
    return self._is_rela