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
def test_display_empty_info(self):
    output = StringIO()
    with redirect_stdout(output):
        res = nsi.display_sysinfo({})
    self.assertIsNone(res)
    output.close()