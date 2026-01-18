from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_aliasedImportShadowModule(self):
    """Imported aliases can shadow the source of the import."""
    self.flakes('from moo import fu as moo; moo')
    self.flakes('import fu as fu; fu')
    self.flakes('import fu.bar as fu; fu')