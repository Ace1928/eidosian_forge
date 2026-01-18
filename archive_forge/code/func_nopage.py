import builtins as builtin_mod
import sys
import types
from pathlib import Path
from . import tools
from IPython.core import page
from IPython.utils import io
from IPython.terminal.interactiveshell import TerminalInteractiveShell
def nopage(strng, start=0, screen_lines=0, pager_cmd=None):
    if isinstance(strng, dict):
        strng = strng.get('text/plain', '')
    print(strng)