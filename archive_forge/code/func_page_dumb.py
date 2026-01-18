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
def page_dumb(strng, start=0, screen_lines=25):
    """Very dumb 'pager' in Python, for when nothing else works.

    Only moves forward, same interface as page(), except for pager_cmd and
    mode.
    """
    if isinstance(strng, dict):
        strng = strng.get('text/plain', '')
    out_ln = strng.splitlines()[start:]
    screens = chop(out_ln, screen_lines - 1)
    if len(screens) == 1:
        print(os.linesep.join(screens[0]))
    else:
        last_escape = ''
        for scr in screens[0:-1]:
            hunk = os.linesep.join(scr)
            print(last_escape + hunk)
            if not page_more():
                return
            esc_list = esc_re.findall(hunk)
            if len(esc_list) > 0:
                last_escape = esc_list[-1]
        print(last_escape + os.linesep.join(screens[-1]))