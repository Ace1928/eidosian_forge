import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_capture_output_logfile_string(self):
    with TempfileManager.new_context() as tempfile:
        logfile = tempfile.create_tempfile()
        self.assertTrue(isinstance(logfile, str))
        with tee.capture_output(logfile):
            print('HELLO WORLD')
        with open(logfile, 'r') as f:
            result = f.read()
        self.assertEqual('HELLO WORLD\n', result)