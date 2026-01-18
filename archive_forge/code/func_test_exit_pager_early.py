import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_exit_pager_early(self) -> None:
    with isolation.isolate(infinite) as env:
        pager = isolation.PagerControl(env)
        self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
        self.assertEqual(MAX_LINES_PER_PAGE, pager.quit())
        self.assertFalse(env.error_output())
    self.assertEqual(141, env.exit_code())