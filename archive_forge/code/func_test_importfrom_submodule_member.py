from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_submodule_member(self):
    binding = ImportationFrom('c', None, 'a.b', 'c')
    assert binding.source_statement == 'from a.b import c'
    assert str(binding) == 'a.b.c'