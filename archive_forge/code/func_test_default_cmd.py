import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_default_cmd(self) -> None:
    with fixtures.EnvironmentVariable('FOO'):
        with fixtures.EnvironmentVariable('BAR'):
            cmd = command.UserSpecifiedPager('FOO', 'BAR')
    self.assertEqual(command.PlatformPager().command(), cmd.command())