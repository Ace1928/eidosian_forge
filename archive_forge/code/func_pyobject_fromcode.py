import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def pyobject_fromcode(self, code, gdbvar=None):
    if gdbvar is not None:
        d = {'varname': gdbvar, 'code': code}
        gdb.execute('set $%(varname)s = %(code)s' % d)
        code = '$' + gdbvar
    return libpython.PyObjectPtr.from_pyobject_ptr(self.get_pyobject(code))