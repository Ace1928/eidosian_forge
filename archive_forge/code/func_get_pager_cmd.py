import os
import io
import re
import sys
import tempfile
import subprocess
from io import UnsupportedOperation
from pathlib import Path
from IPython import get_ipython
from IPython.display import display
from IPython.core.error import TryNext
from IPython.utils.data import chop
from IPython.utils.process import system
from IPython.utils.terminal import get_terminal_size
from IPython.utils import py3compat
def get_pager_cmd(pager_cmd=None):
    """Return a pager command.

    Makes some attempts at finding an OS-correct one.
    """
    if os.name == 'posix':
        default_pager_cmd = 'less -R'
    elif os.name in ['nt', 'dos']:
        default_pager_cmd = 'type'
    if pager_cmd is None:
        try:
            pager_cmd = os.environ['PAGER']
        except:
            pager_cmd = default_pager_cmd
    if pager_cmd == 'less' and '-r' not in os.environ.get('LESS', '').lower():
        pager_cmd += ' -R'
    return pager_cmd