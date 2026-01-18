import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_failsWithEmptyFQPN(self):
    """
        L{wrapFQPN} raises L{InvalidFQPN} when given an empty string.
        """
    with self.assertRaises(self.InvalidFQPN):
        self.wrapFQPN('')