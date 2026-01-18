import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def get_repr(self, pyobject):
    return pyobject.get_truncated_repr(libpython.MAX_OUTPUT_LEN)