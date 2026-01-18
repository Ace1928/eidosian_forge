from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_import_submodule(self):
    binding = SubmoduleImportation('a.b', None)
    assert binding.source_statement == 'import a.b'
    assert str(binding) == 'a.b'