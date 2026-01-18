import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_ignoresImportedClass(self):
    """
        When given a L{twisted.python.modules.PythonAttribute} that refers
        to a class imported from another module, any
        L{MethodicalMachine}s on that class are ignored.

        This behavior ensures that a machine is only discovered on a
        class when visiting the module where that class was defined.
        """
    originalSource = '\n        from automat import MethodicalMachine\n\n        class PythonClass(object):\n            _classMachine = MethodicalMachine()\n        '
    importingSource = '\n        from original import PythonClass\n        '
    self.makeModule(originalSource, self.pathDir, 'original.py')
    importingModule = self.makeModule(importingSource, self.pathDir, 'importing.py')
    self.assertFalse(list(self.findMachinesViaWrapper(importingModule)))