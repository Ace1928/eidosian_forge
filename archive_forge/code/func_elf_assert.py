from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
def elf_assert(cond, msg=''):
    """ Assert that cond is True, otherwise raise ELFError(msg)
    """
    _assert_with_exception(cond, msg, ELFError)