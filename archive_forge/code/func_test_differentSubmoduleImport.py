from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_differentSubmoduleImport(self):
    """
        If two different submodules of a package are imported, no duplicate
        import warning is reported for the package.
        """
    self.flakes('\n        import fu.bar, fu.baz\n        fu.bar, fu.baz\n        ')
    self.flakes('\n        import fu.bar\n        import fu.baz\n        fu.bar, fu.baz\n        ')