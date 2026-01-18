import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_cygwin_cmd(self) -> None:
    with PlatformFixture('cygwin'):
        cmd = command.PlatformPager()
        self.assertEqual(['less'], cmd.command())