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
def test_gettempdir(self):
    context = self.TM.push()
    fname = context.create_tempfile()
    self.assertIsInstance(fname, str)
    system_tmpdir = os.path.dirname(fname)
    self.assertEqual(system_tmpdir, tempfile.gettempdir())
    tmpdir = context.gettempdir()
    self.assertIsInstance(tmpdir, str)
    self.assertEqual(tmpdir, system_tmpdir)
    tmpdirb = context.gettempdirb()
    self.assertIsInstance(tmpdirb, bytes)
    self.assertEqual(tmpdirb.decode(), tmpdir)
    manager_tmpdir = context.create_tempdir()
    self.assertNotEqual(manager_tmpdir, system_tmpdir)
    self.TM.tempdir = manager_tmpdir
    fname = context.create_tempfile()
    self.assertIsInstance(fname, str)
    tmpdir = context.gettempdir()
    self.assertIsInstance(tmpdir, str)
    self.assertEqual(tmpdir, manager_tmpdir)
    tmpdirb = context.gettempdirb()
    self.assertIsInstance(tmpdirb, bytes)
    self.assertEqual(tmpdirb.decode(), tmpdir)
    context_tmpdir = context.create_tempdir()
    self.assertNotEqual(context_tmpdir, system_tmpdir)
    self.assertNotEqual(context_tmpdir, manager_tmpdir)
    context.tempdir = context_tmpdir
    fname = context.create_tempfile()
    self.assertIsInstance(fname, str)
    tmpdir = context.gettempdir()
    self.assertIsInstance(tmpdir, str)
    self.assertEqual(tmpdir, context_tmpdir)
    tmpdirb = context.gettempdirb()
    self.assertIsInstance(tmpdirb, bytes)
    self.assertEqual(tmpdirb.decode(), tmpdir)
    context.tempdir = context_tmpdir.encode()
    fname = context.create_tempfile()
    self.assertIsInstance(fname, bytes)
    tmpdir = context.gettempdir()
    self.assertIsInstance(tmpdir, str)
    self.assertEqual(tmpdir, context_tmpdir)
    tmpdirb = context.gettempdirb()
    self.assertIsInstance(tmpdirb, bytes)
    self.assertEqual(tmpdirb.decode(), tmpdir)
    self.TM.pop()