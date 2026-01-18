from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_member(self):
    binding = ImportationFrom('b', None, 'a', 'b')
    assert binding.source_statement == 'from a import b'
    assert str(binding) == 'a.b'