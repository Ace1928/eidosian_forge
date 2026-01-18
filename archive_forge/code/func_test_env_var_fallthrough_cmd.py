import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_env_var_fallthrough_cmd(self) -> None:
    with fixtures.EnvironmentVariable('BAR', 'bar'):
        cmd = command.UserSpecifiedPager('FOO', 'BAR')
    self.assertEqual(['bar'], cmd.command())