from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classNameUndefinedInClassBody(self):
    """
        If a class name is used in the body of that class's definition and
        the name is not already defined, a warning is emitted.
        """
    self.flakes('\n        class foo:\n            foo\n        ', m.UndefinedName)