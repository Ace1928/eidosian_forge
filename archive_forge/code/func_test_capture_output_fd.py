import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_capture_output_fd(self):
    r, w = os.pipe()
    os.dup2(w, 1)
    sys.stdout = os.fdopen(1, 'w', closefd=False)
    with tee.capture_output(capture_fd=True) as OUT:
        sys.stdout.write('to_stdout_1\n')
        sys.stdout.flush()
        with os.fdopen(1, 'w', closefd=False) as F:
            F.write('to_fd1_1\n')
            F.flush()
    sys.stdout.write('to_stdout_2\n')
    sys.stdout.flush()
    with os.fdopen(1, 'w', closefd=False) as F:
        F.write('to_fd1_2\n')
        F.flush()
    self.assertEqual(OUT.getvalue(), 'to_stdout_1\nto_fd1_1\n')
    with os.fdopen(r, 'r') as FILE:
        os.close(1)
        os.close(w)
        self.assertEqual(FILE.read(), 'to_stdout_2\nto_fd1_2\n')