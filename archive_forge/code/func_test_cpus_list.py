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
@skipUnless(platform.system() in ('Linux', 'Windows'), 'CPUs allowed info only available on Linux and Windows')
def test_cpus_list(self):
    self.assertEqual(self.info[nsi._cpus_allowed], len(self.cpus_list))
    self.assertEqual(self.info[nsi._cpus_list], ' '.join((str(n) for n in self.cpus_list)))