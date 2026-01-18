from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assert_with_message(self):
    """An assert with a message is not an error."""
    self.flakes("\n        a = 1\n        assert a, 'x'\n        ")