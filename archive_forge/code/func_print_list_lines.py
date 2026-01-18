import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def print_list_lines(self, filename, first, last):
    """The printing (as opposed to the parsing part of a 'list'
        command."""
    try:
        Colors = self.color_scheme_table.active_colors
        ColorsNormal = Colors.Normal
        tpl_line = '%%s%s%%s %s%%s' % (Colors.lineno, ColorsNormal)
        tpl_line_em = '%%s%s%%s %s%%s%s' % (Colors.linenoEm, Colors.line, ColorsNormal)
        src = []
        if filename == '<string>' and hasattr(self, '_exec_filename'):
            filename = self._exec_filename
        for lineno in range(first, last + 1):
            line = linecache.getline(filename, lineno)
            if not line:
                break
            if lineno == self.curframe.f_lineno:
                line = self.__format_line(tpl_line_em, filename, lineno, line, arrow=True)
            else:
                line = self.__format_line(tpl_line, filename, lineno, line, arrow=False)
            src.append(line)
            self.lineno = lineno
        print(''.join(src), file=self.stdout)
    except KeyboardInterrupt:
        pass