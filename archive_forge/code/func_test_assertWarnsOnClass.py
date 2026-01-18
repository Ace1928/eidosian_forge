import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsOnClass(self):
    """
        Test assertWarns works when creating a class instance.
        """

    class Warn:

        def __init__(self):
            warnings.warn('Do not call me', category=RuntimeWarning)
    r = self.assertWarns(RuntimeWarning, 'Do not call me', __file__, Warn)
    self.assertTrue(isinstance(r, Warn))
    r = self.assertWarns(RuntimeWarning, 'Do not call me', __file__, Warn)
    self.assertTrue(isinstance(r, Warn))