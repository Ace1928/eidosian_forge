import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def test_check_shard(self):
    tmpAll = self.get_testsuite_listing([])
    tmp1 = self.get_testsuite_listing(['-j', '0:2'])
    tmp2 = self.get_testsuite_listing(['-j', '1:2'])
    lAll = set(self._get_numba_tests_from_listing(tmpAll))
    l1 = set(self._get_numba_tests_from_listing(tmp1))
    l2 = set(self._get_numba_tests_from_listing(tmp2))
    self.assertLess(abs(len(l2) - len(l1)), len(lAll) / 20)
    self.assertLess(len(l1), len(lAll))
    self.assertLess(len(l2), len(lAll))