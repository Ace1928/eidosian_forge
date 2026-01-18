import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def test_frame_type(self):
    frame = self.pyobject_fromcode('PyEval_GetFrame()')
    self.assertEqual(type(frame), libpython.PyFrameObjectPtr)