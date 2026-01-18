from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def initial_length_field_size(self):
    """ Size of an initial length field.
        """
    return 4 if self.dwarf_format == 32 else 12