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
def test_del_clears_contexts(self):
    TM = TempfileManagerClass()
    ctx = TM.push()
    fname = ctx.create_tempfile()
    self.assertTrue(os.path.exists(fname))
    with LoggingIntercept() as LOG:
        TM = None
        gc.collect()
        gc.collect()
        gc.collect()
    self.assertFalse(os.path.exists(fname))
    self.assertEqual(LOG.getvalue().strip(), 'Temporary files created through TempfileManager contexts have not been deleted (observed during TempfileManager instance shutdown).\nUndeleted entries:\n\t%s\nTempfileManagerClass instance: un-popped tempfile contexts still exist during TempfileManager instance shutdown' % fname)