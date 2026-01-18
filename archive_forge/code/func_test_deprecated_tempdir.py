import gc
import glob
import os
import shutil
import sys
import tempfile
from io import StringIO
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.common.tempfiles as tempfiles
from pyomo.common.dependencies import pyutilib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import (
@unittest.skipUnless(pyutilib_available, 'deprecation test requires pyutilib')
def test_deprecated_tempdir(self):
    self.TM.push()
    try:
        tmpdir = self.TM.create_tempdir()
        _orig = tempfiles.pyutilib_tempfiles.TempfileManager.tempdir
        tempfiles.pyutilib_tempfiles.TempfileManager.tempdir = tmpdir
        self.TM.tempdir = None
        with LoggingIntercept() as LOG:
            fname = self.TM.create_tempfile()
        self.assertIn('The use of the PyUtilib TempfileManager.tempdir to specify the default location for Pyomo temporary files', LOG.getvalue().replace('\n', ' '))
        with LoggingIntercept() as LOG:
            dname = self.TM.create_tempdir()
        self.assertIn('The use of the PyUtilib TempfileManager.tempdir to specify the default location for Pyomo temporary files', LOG.getvalue().replace('\n', ' '))
    finally:
        self.TM.pop()
        tempfiles.pyutilib_tempfiles.TempfileManager.tempdir = _orig