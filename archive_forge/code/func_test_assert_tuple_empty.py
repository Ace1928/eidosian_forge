from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assert_tuple_empty(self):
    """An assert of an empty tuple is always False."""
    self.flakes('\n        assert ()\n        ')