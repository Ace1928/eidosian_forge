import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_interrupt_in_middle_after_complete(self) -> None:
    num_lines = 100
    with isolation.isolate(finite(num_lines)) as env:
        pager = isolation.PagerControl(env)
        self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
        for i in range(100):
            env.interrupt()
        self.assertEqual(MAX_LINES_PER_PAGE, pager.quit())
        self.assertFalse(env.error_output())
    self.assertEqual(0, env.exit_code())