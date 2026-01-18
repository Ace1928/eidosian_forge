from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors_func_call(self):
    self.flakes('\n        def test(thing):\n            pass\n        test({True: 1, None: 2, False: 1})\n        ')