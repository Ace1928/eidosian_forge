from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_future(self):
    binding = FutureImportation('print_function', None, None)
    assert binding.source_statement == 'from __future__ import print_function'
    assert str(binding) == '__future__.print_function'