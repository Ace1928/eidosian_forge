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
def test_content_type(self):
    for t, keys in self.contents.items():
        for k in keys:
            with self.subTest(k=k):
                self.assertIsInstance(self.info[k], t)