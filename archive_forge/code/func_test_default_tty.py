import unittest
import fixtures  # type: ignore
from autopage.tests import sinks
import autopage
def test_default_tty(self) -> None:
    with sinks.TTYFixture() as inp:
        with fixtures.MonkeyPatch('sys.stdin', inp.stream):
            self.assertTrue(autopage.line_buffer_from_input())