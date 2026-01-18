import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader
def get_default_prog_name(self, cli: 'BaseCommand') -> str:
    """Given a command object it will return the default program name
        for it.  The default is the `name` attribute or ``"root"`` if not
        set.
        """
    return cli.name or 'root'