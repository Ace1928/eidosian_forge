from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_non_traceback_param(self):
    with self.assertRaises(TypeError) as exc:
        _test_extension.test_throw_exact(greenlet.getcurrent(), Exception, Exception(), self)
    self.assertEqual(str(exc.exception), 'throw() third argument must be a traceback object')