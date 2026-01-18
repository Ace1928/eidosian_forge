from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assignToMember(self):
    """
        Assigning to a member of another object and then not using that member
        variable is perfectly acceptable. Do not mistake it for an unused
        local variable.
        """
    self.flakes('\n        class b:\n            pass\n        def a():\n            b.foo = 1\n        ')