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
def test_mkstemp(self):
    context = self.TM.new_context()
    fd, fname = context.mkstemp('suffix', 'prefix')
    self.assertRegex(os.path.basename(fname), '^prefix')
    self.assertRegex(os.path.basename(fname), 'suffix$')
    self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
    self.assertTrue(os.path.exists(fname))
    self.assertTrue(os.path.isfile(fname))
    os.fsync(fd)
    context.release()
    self.assertFalse(os.path.exists(fname))
    with self.assertRaises(OSError):
        os.fsync(fd)
    context = self.TM.new_context()
    fd, fname = context.mkstemp('suffix', 'prefix')
    os.close(fd)
    context.release()