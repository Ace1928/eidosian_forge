from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_keys_tuples_same_first_element(self):
    self.flakes('{(0,1): 1, (0,2): 1}')