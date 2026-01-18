from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_star_relative(self):
    binding = StarImportation('.b', None)
    assert binding.source_statement == 'from .b import *'
    assert str(binding) == '.b.*'