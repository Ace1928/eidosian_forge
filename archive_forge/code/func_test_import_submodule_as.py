from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_import_submodule_as(self):
    binding = Importation('c', None, 'a.b')
    assert binding.source_statement == 'import a.b as c'
    assert str(binding) == 'a.b as c'