import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_interrupt_early(self) -> None:
    with isolation.isolate(infinite, stdout_pipe=True) as env:
        env.interrupt()
        with env.stdout_pipe() as out:
            output = out.readlines()
        self.assertGreater(len(output), 0)
        self.assertFalse(env.error_output())
    self.assertEqual(130, env.exit_code())