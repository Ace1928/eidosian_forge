from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame()
def get_source_desc(self, frame):
    filename = lineno = lexer = None
    if self.is_cython_function(frame):
        filename = self.get_cython_function(frame).module.filename
        filename_and_lineno = self.get_cython_lineno(frame)
        assert filename == filename_and_lineno[0]
        lineno = filename_and_lineno[1]
        if pygments:
            lexer = pygments.lexers.CythonLexer(stripall=False)
    elif self.is_python_function(frame):
        pyframeobject = libpython.Frame(frame).get_pyop()
        if not pyframeobject:
            raise gdb.GdbError('Unable to read information on python frame')
        filename = pyframeobject.filename()
        lineno = pyframeobject.current_line_num()
        if pygments:
            lexer = pygments.lexers.PythonLexer(stripall=False)
    else:
        symbol_and_line_obj = frame.find_sal()
        if not symbol_and_line_obj or not symbol_and_line_obj.symtab:
            filename = None
            lineno = 0
        else:
            filename = symbol_and_line_obj.symtab.fullname()
            lineno = symbol_and_line_obj.line
            if pygments:
                lexer = pygments.lexers.CLexer(stripall=False)
    return (SourceFileDescriptor(filename, lexer), lineno)