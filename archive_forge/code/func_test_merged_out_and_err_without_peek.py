import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_merged_out_and_err_without_peek(self):
    a = StringIO()
    b = StringIO()
    try:
        _tmp, tee._peek_available = (tee._peek_available, False)
        with tee.TeeStream(a, b) as t:
            t.STDOUT
            t.STDERR
            t.STDERR.write('Hello\n')
            t.STDERR.flush()
            time.sleep(tee._poll_interval * 2)
            t.STDOUT.write('World\n')
    finally:
        tee._peek_available = _tmp
    self.assertEqual(a.getvalue(), 'Hello\nWorld\n')
    self.assertEqual(b.getvalue(), 'Hello\nWorld\n')