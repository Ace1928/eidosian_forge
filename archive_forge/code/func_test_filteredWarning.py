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
def test_filteredWarning(self):
    """
        L{deprecate.warnAboutFunction} emits a warning that will be filtered if
        L{warnings.filterwarning} is called with the module name of the
        deprecated function.
        """
    del warnings.filters[:]
    warnings.filterwarnings(action='ignore', module='twisted_private_helper')
    from twisted_private_helper import module
    module.callTestFunction()
    warningsShown = self.flushWarnings()
    self.assertEqual(len(warningsShown), 0)