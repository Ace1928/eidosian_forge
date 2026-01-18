from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assert_without_message(self):
    """An assert without a message is not an error."""
    self.flakes('\n        a = 1\n        assert a\n        ')