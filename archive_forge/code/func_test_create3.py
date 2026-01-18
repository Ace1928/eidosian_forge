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
def test_create3(self):
    """Test create logic - no options"""
    fname = TempfileManager.create_tempfile(suffix='bar')
    OUTPUT = open(fname, 'w')
    OUTPUT.write('tempfile\n')
    OUTPUT.close()
    self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
    fname = os.path.basename(fname)
    self.assertTrue(fname.endswith('bar'))