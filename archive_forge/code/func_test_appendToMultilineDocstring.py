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
def test_appendToMultilineDocstring(self):
    """
        Appending to a multi-line docstring places the messade on a new line,
        with a blank line separating it from the rest of the docstring.

        Because we have multiple lines, we have enough information to do
        indentation.
        """

    def multiLineDocstring():
        """
            This is a multi-line docstring.
            """

    def expectedDocstring():
        """
            This is a multi-line docstring.

            Appended text.
            """
    _appendToDocstring(multiLineDocstring, 'Appended text.')
    self.assertEqual(expectedDocstring.__doc__, multiLineDocstring.__doc__)