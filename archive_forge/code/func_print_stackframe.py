from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame(err=False)
def print_stackframe(self, frame, index, is_c=False):
    """
        Print a C, Cython or Python stack frame and the line of source code
        if available.
        """
    selected_frame = gdb.selected_frame()
    frame.select()
    try:
        source_desc, lineno = self.get_source_desc(frame)
    except NoFunctionNameInFrameError:
        print('#%-2d Unknown Frame (compile with -g)' % index)
        return
    if not is_c and self.is_python_function(frame):
        pyframe = libpython.Frame(frame).get_pyop()
        if pyframe is None or pyframe.is_optimized_out():
            return self.print_stackframe(frame, index, is_c=True)
        func_name = pyframe.co_name
        func_cname = 'PyEval_EvalFrameEx'
        func_args = []
    elif self.is_cython_function(frame):
        cyfunc = self.get_cython_function(frame)
        f = lambda arg: self.cy.cy_cvalue.invoke(arg, frame=frame)
        func_name = cyfunc.name
        func_cname = cyfunc.cname
        func_args = []
    else:
        source_desc, lineno = self.get_source_desc(frame)
        func_name = frame.name()
        func_cname = func_name
        func_args = []
    try:
        gdb_value = gdb.parse_and_eval(func_cname)
    except RuntimeError:
        func_address = 0
    else:
        func_address = gdb_value.address
        if not isinstance(func_address, int):
            if not isinstance(func_address, (str, bytes)):
                func_address = str(func_address)
            func_address = int(func_address.split()[0], 0)
    a = ', '.join(('%s=%s' % (name, val) for name, val in func_args))
    sys.stdout.write('#%-2d 0x%016x in %s(%s)' % (index, func_address, func_name, a))
    if source_desc.filename is not None:
        sys.stdout.write(' at %s:%s' % (source_desc.filename, lineno))
    sys.stdout.write('\n')
    try:
        sys.stdout.write('    ' + source_desc.get_source(lineno))
    except gdb.GdbError:
        pass
    selected_frame.select()