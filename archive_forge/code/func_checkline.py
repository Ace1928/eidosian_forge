import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def checkline(self, filename, lineno):
    """Check whether specified line seems to be executable.

        Return `lineno` if it is, 0 if not (e.g. a docstring, comment, blank
        line or EOF). Warning: testing is not comprehensive.
        """
    frame = getattr(self, 'curframe', None)
    globs = frame.f_globals if frame else None
    line = linecache.getline(filename, lineno, globs)
    if not line:
        self.message('End of file')
        return 0
    line = line.strip()
    if not line or line[0] == '#' or line[:3] == '"""' or (line[:3] == "'''"):
        self.error('Blank or comment')
        return 0
    return lineno