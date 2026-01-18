from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_submodule_member_as(self):
    binding = ImportationFrom('d', None, 'a.b', 'c')
    assert binding.source_statement == 'from a.b import c as d'
    assert str(binding) == 'a.b.c as d'