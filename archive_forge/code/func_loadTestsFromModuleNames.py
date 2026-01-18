import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def loadTestsFromModuleNames(self, names):
    """use a custom means to load tests from modules.

        There is an undesirable glitch in the python TestLoader where a
        import error is ignore. We think this can be solved by ensuring the
        requested name is resolvable, if its not raising the original error.
        """
    result = self.suiteClass()
    for name in names:
        result.addTests(self.loadTestsFromModuleName(name))
    return result