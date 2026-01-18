import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def test_check_sharding_equivalent(self):
    sharded = list()
    for i in range(3):
        subset = self.get_testsuite_listing(['-j', '{}:3'.format(i)])
        slist = [*self._get_numba_tests_from_listing(subset)]
        sharded.append(slist)
    tmp = self.get_testsuite_listing(['--tag', 'always_test'])
    always_running = set(self._get_numba_tests_from_listing(tmp))
    self.assertGreaterEqual(len(always_running), 1)
    sharded_sets = [set(x) for x in sharded]
    for i in range(len(sharded)):
        self.assertEqual(len(sharded_sets[i]), len(sharded[i]))
    for shard in sharded_sets:
        for test in always_running:
            self.assertIn(test, shard)
            shard.remove(test)
            self.assertNotIn(test, shard)
    for a, b in itertools.combinations(sharded_sets, 2):
        self.assertFalse(a & b)
    sum_of_parts = set()
    for x in sharded_sets:
        sum_of_parts.update(x)
    sum_of_parts.update(always_running)
    full_listing = set(self._get_numba_tests_from_listing(self.get_testsuite_listing([])))
    self.assertEqual(sum_of_parts, full_listing)