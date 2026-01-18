import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_interrupt_early_quit(self) -> None:
    with isolation.isolate(infinite) as env:
        pager = isolation.PagerControl(env)
        self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
        env.interrupt()
        pager.quit()
        self.assertGreater(pager.total_lines(), MAX_LINES_PER_PAGE)
        self.assertFalse(env.error_output())
    self.assertEqual(130, env.exit_code())