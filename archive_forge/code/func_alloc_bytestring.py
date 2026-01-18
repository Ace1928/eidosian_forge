import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def alloc_bytestring(self, string, gdbvar=None):
    if inferior_python_version < (3, 0):
        funcname = 'PyString_FromStringAndSize'
    else:
        funcname = 'PyBytes_FromStringAndSize'
    assert b'"' not in string
    code = '(PyObject *) %s("%s", %d)' % (funcname, string.decode('iso8859-1'), len(string))
    return self.pyobject_fromcode(code, gdbvar=gdbvar)