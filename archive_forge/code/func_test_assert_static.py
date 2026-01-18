from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assert_static(self):
    """An assert of a static value is not an error."""
    self.flakes('\n        assert True\n        assert 1\n        ')