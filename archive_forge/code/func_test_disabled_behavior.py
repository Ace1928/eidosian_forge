import unittest
from numba import njit
from numba.tests.support import TestCase, override_config
from numba.misc import llvm_pass_timings as lpt
def test_disabled_behavior(self):

    @njit
    def foo(n):
        c = 0
        for i in range(n):
            c += i
        return c
    with override_config('LLVM_PASS_TIMINGS', False):
        foo(10)
    md = foo.get_metadata(foo.signatures[0])
    timings = md['llvm_pass_timings']
    self.assertEqual(timings.summary(), 'No pass timings were recorded')
    self.assertIsNone(timings.get_total_time())
    self.assertEqual(timings.list_longest_first(), [])