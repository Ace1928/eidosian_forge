import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_interrupt_at_end_after_complete(self) -> None:
    num_lines = 100
    with isolation.isolate(finite(num_lines)) as env:
        pager = isolation.PagerControl(env)
        while pager.advance():
            continue
        self.assertEqual(num_lines, pager.total_lines())
        for i in range(100):
            env.interrupt()
        self.assertEqual(0, pager.quit())
        self.assertFalse(env.error_output())
    self.assertEqual(0, env.exit_code())