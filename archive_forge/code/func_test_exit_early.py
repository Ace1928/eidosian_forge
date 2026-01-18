import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_exit_early(self) -> None:
    with isolation.isolate(infinite, stdout_pipe=True) as env:
        with env.stdout_pipe() as out:
            for i in range(500):
                out.readline()
        self.assertFalse(env.error_output())
    self.assertEqual(141, env.exit_code())