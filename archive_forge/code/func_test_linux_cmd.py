import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_linux_cmd(self) -> None:
    with PlatformFixture('linux'):
        cmd = command.PlatformPager()
        self.assertEqual(['less'], cmd.command())