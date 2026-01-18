import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_failsWithNoModule(self):
    """
        L{isOriginalLocation} returns False when the attribute refers to an
        object whose source module cannot be determined.
        """
    source = '        class Fake(object):\n            pass\n        hasEmptyModule = Fake()\n        hasEmptyModule.__module__ = None\n        '
    moduleDict = self.makeModuleAsDict(source, self.pathDir, 'empty_module_attr.py')
    self.assertFalse(self.isOriginalLocation(moduleDict['empty_module_attr.hasEmptyModule']))