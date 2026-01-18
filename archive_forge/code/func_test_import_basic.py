from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_import_basic(self):
    binding = Importation('a', None, 'a')
    assert binding.source_statement == 'import a'
    assert str(binding) == 'a'