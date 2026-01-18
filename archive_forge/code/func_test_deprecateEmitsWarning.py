import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def test_deprecateEmitsWarning(self):
    """
        Decorating a callable with L{deprecated} emits a warning.
        """
    version = Version('Twisted', 8, 0, 0)
    dummy = deprecated(version)(dummyCallable)

    def addStackLevel():
        dummy()
    with catch_warnings(record=True) as caught:
        simplefilter('always')
        addStackLevel()
        self.assertEqual(caught[0].category, DeprecationWarning)
        self.assertEqual(str(caught[0].message), getDeprecationWarningString(dummyCallable, version))
        self.assertEqual(caught[0].filename.rstrip('co'), __file__.rstrip('co'))