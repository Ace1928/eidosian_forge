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
def test_appendToSingleLineDocstring(self):
    """
        Appending to a single line docstring places the message on a new line,
        with a blank line separating it from the rest of the docstring.

        The docstring ends with a newline, conforming to Twisted and PEP 8
        standards. Unfortunately, the indentation is incorrect, since the
        existing docstring doesn't have enough info to help us indent
        properly.
        """

    def singleLineDocstring():
        """This doesn't comply with standards, but is here for a test."""
    _appendToDocstring(singleLineDocstring, 'Appended text.')
    self.assertEqual(["This doesn't comply with standards, but is here for a test.", '', 'Appended text.'], singleLineDocstring.__doc__.splitlines())
    self.assertTrue(singleLineDocstring.__doc__.endswith('\n'))