import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
@staticmethod
def strip_last_newline(str):
    """Strip one last newline, if present.

        >>> CommandLineTool.strip_last_newline('foo')
        'foo'
        >>> CommandLineTool.strip_last_newline('foo\\n')
        'foo'
        """
    slc = slice(-1 if str.endswith('\n') else None)
    return str[slc]