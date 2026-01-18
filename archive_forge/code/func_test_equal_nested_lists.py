from unittest import TestCase
from jsonschema._utils import equal
def test_equal_nested_lists(self):
    list_1 = ['a', ['b', 'c'], 'd']
    list_2 = ['a', ['b', 'c'], 'd']
    self.assertTrue(equal(list_1, list_2))