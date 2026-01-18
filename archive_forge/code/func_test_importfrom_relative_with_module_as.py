from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_relative_with_module_as(self):
    binding = ImportationFrom('c', None, '..a', 'b')
    assert binding.source_statement == 'from ..a import b as c'
    assert str(binding) == '..a.b as c'