import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_duplicate_capture_output(self):
    out = StringIO()
    capture = tee.capture_output(out)
    capture.setup()
    try:
        with self.assertRaisesRegex(RuntimeError, 'Duplicate call to capture_output.setup'):
            capture.setup()
    finally:
        capture.reset()