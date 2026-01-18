from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_checksCurrentDirectory(self):
    """
        L{tool} adds '' to sys.path to ensure
        L{automat._discover.findMachines} searches the current
        directory.
        """
    self.tool(argv=[self.fakeFQPN])
    self.assertEqual(self.fakeSysPath[0], '')