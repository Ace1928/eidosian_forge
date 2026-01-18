from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors_instance_attributes(self):
    self.flakes('\n        class Test():\n            pass\n        f = Test()\n        f.a = 1\n        {f.a: 1, f.a: 1}\n        ')