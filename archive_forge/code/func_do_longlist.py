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
def do_longlist(self, arg):
    """longlist | ll
        List the whole source code for the current function or frame.
        """
    filename = self.curframe.f_code.co_filename
    breaklist = self.get_file_breaks(filename)
    try:
        lines, lineno = self._getsourcelines(self.curframe)
    except OSError as err:
        self.error(err)
        return
    self._print_lines(lines, lineno, breaklist, self.curframe)