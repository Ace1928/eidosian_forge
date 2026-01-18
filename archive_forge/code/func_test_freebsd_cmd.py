import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_freebsd_cmd(self) -> None:
    with PlatformFixture('freebsd8'):
        cmd = command.PlatformPager()
        self.assertEqual(['less'], cmd.command())