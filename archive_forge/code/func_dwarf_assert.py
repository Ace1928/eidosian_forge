from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
def dwarf_assert(cond, msg=''):
    """ Assert that cond is True, otherwise raise DWARFError(msg)
    """
    _assert_with_exception(cond, msg, DWARFError)