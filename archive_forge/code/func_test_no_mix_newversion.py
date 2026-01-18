from __future__ import division, absolute_import
import sys
import os
import datetime
from twisted.python.filepath import FilePath
from twisted.python.compat import NativeStringIO
from twisted.trial.unittest import TestCase
from incremental.update import _run, run
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
def test_no_mix_newversion(self):
    """
        The `--newversion` flag can't be mixed with --patch, --rc, --post,
        or --dev.
        """
    out = []
    with self.assertRaises(ValueError) as e:
        _run(u'inctestpkg', path=None, newversion='1', patch=True, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(e.exception.args[0], 'Only give --newversion')
    with self.assertRaises(ValueError) as e:
        _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(e.exception.args[0], 'Only give --newversion')
    with self.assertRaises(ValueError) as e:
        _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=False, post=True, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(e.exception.args[0], 'Only give --newversion')
    with self.assertRaises(ValueError) as e:
        _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(e.exception.args[0], 'Only give --newversion')