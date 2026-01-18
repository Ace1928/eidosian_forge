import platform
import unittest
from unittest import skipUnless
from unittest.mock import NonCallableMock
from itertools import chain
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
from numba.tests.support import TestCase
import numba.misc.numba_sysinfo as nsi
def test_has_correct_values(self):
    self.assertEqual(self.info[nsi._mem_total], self.mem_total)
    self.assertEqual(self.info[nsi._mem_available], self.mem_available)