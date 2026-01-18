import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_aix_cmd(self) -> None:
    with PlatformFixture('aix'):
        cmd = command.PlatformPager()
        self.assertEqual(['more'], cmd.command())
    with PlatformFixture('aix7'):
        cmd = command.PlatformPager()
        self.assertEqual(['more'], cmd.command())