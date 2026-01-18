import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_exit_early_buffered(self) -> None:
    num_lines = 10
    with isolation.isolate(from_stdin, stdin_pipe=True, stdout_pipe=True) as env:
        with env.stdin_pipe() as in_pipe:
            for i in range(num_lines):
                print(i, file=in_pipe)
        with env.stdout_pipe():
            pass
        self.assertFalse(env.error_output())
    self.assertEqual(141, env.exit_code())