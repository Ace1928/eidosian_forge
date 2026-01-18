import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_binary_tee(self):
    a = BytesIO()
    b = BytesIO()
    with tee.TeeStream(a, b) as t:
        t.open('wb').write(b'Hello\n')
    self.assertEqual(a.getvalue(), b'Hello\n')
    self.assertEqual(b.getvalue(), b'Hello\n')