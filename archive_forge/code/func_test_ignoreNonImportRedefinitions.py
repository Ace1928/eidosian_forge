from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_ignoreNonImportRedefinitions(self):
    self.flakes('a = 1; a = 2')