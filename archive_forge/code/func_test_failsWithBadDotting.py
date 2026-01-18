import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_failsWithBadDotting(self):
    """"
        L{wrapFQPN} raises L{InvalidFQPN} when given a badly-dotted
        FQPN.  (e.g., x..y).
        """
    for bad in ('.fails', 'fails.', 'this..fails'):
        with self.assertRaises(self.InvalidFQPN):
            self.wrapFQPN(bad)