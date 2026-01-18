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
def test_add_tempfile(self):
    context1 = self.TM.push()
    context2 = self.TM.push()
    fname = context1.create_tempfile()
    dname = context1.create_tempdir()
    sub_fname = os.path.join(dname, 'testfile')
    self.TM.add_tempfile(fname)
    with self.assertRaisesRegex(IOError, 'Temporary file does not exist: %s' % sub_fname.replace('\\', '\\\\')):
        self.TM.add_tempfile(sub_fname)
    self.TM.add_tempfile(sub_fname, exists=False)
    with open(sub_fname, 'w') as FILE:
        FILE.write('\n')
    self.assertTrue(os.path.exists(fname))
    self.assertTrue(os.path.exists(dname))
    self.assertTrue(os.path.exists(sub_fname))
    self.TM.pop()
    self.assertFalse(os.path.exists(fname))
    self.assertTrue(os.path.exists(dname))
    self.assertFalse(os.path.exists(sub_fname))
    self.TM.pop()
    self.assertFalse(os.path.exists(fname))
    self.assertFalse(os.path.exists(dname))
    self.assertFalse(os.path.exists(sub_fname))