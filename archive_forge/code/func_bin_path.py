import doctest
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import ClassVar, List
from unittest import SkipTest, expectedFailure, skipIf
from unittest import TestCase as _TestCase
def bin_path(self, name):
    """Determine the full path of a binary.

        Args:
          name: Name of the script
        Returns: Full path
        """
    for d in self.bin_directories:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    else:
        raise SkipTest('Unable to find binary %s' % name)