import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
@classmethod
def pass_from_pipe(cls):
    """Return password from pipe if not on TTY, else False."""
    is_pipe = not sys.stdin.isatty()
    return is_pipe and cls.strip_last_newline(sys.stdin.read())