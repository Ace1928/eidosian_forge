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
def test_filteredOnceWarning(self):
    """
        L{deprecate.warnAboutFunction} emits a warning that will be filtered
        once if L{warnings.filterwarning} is called with the module name of the
        deprecated function and an action of once.
        """
    del warnings.filters[:]
    warnings.filterwarnings(action='module', module='twisted_private_helper')
    from twisted_private_helper import module
    module.callTestFunction()
    module.callTestFunction()
    warningsShown = self.flushWarnings()
    self.assertEqual(len(warningsShown), 1)
    message = warningsShown[0]['message']
    category = warningsShown[0]['category']
    filename = warningsShown[0]['filename']
    lineno = warningsShown[0]['lineno']
    msg = warnings.formatwarning(message, category, filename, lineno)
    self.assertTrue(msg.endswith('module.py:9: DeprecationWarning: A Warning String\n  return a\n'), f'Unexpected warning string: {msg!r}')