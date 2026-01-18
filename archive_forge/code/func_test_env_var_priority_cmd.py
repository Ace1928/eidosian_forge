import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_env_var_priority_cmd(self) -> None:
    with fixtures.EnvironmentVariable('FOO', 'foo'):
        cmd = command.UserSpecifiedPager('FOO', 'BAR')
    self.assertEqual(['foo'], cmd.command())