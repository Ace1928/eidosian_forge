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
def simpleModuleEntry(self):
    """
        Add a sample module and package to the path, returning a L{FilePath}
        pointing at the module which will be loadable as C{package.module}.
        """
    paths = self.pathEntryTree({b'package': {b'__init__.py': self._packageInit.encode('utf-8'), b'module.py': b''}})
    return paths[b'package'][b'module.py']