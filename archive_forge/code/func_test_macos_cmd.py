import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_macos_cmd(self) -> None:
    with PlatformFixture('darwin'):
        cmd = command.PlatformPager()
        self.assertEqual(['less'], cmd.command())