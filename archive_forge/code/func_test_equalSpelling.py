import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_equalSpelling(self):
    for name, value in vars(self).items():
        if not callable(value):
            continue
        if name.endswith('Equal'):
            self.assertTrue(hasattr(self, name + 's'), f'{name} but no {name}s')
            self.assertEqual(value, getattr(self, name + 's'))
        if name.endswith('Equals'):
            self.assertTrue(hasattr(self, name[:-1]), f'{name} but no {name[:-1]}')
            self.assertEqual(value, getattr(self, name[:-1]))