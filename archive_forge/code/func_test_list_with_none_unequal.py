from unittest import TestCase
from jsonschema._utils import equal
def test_list_with_none_unequal(self):
    list_1 = ['a', 'b', None]
    list_2 = ['a', 'b', 'c']
    self.assertFalse(equal(list_1, list_2))
    list_1 = ['a', 'b', None]
    list_2 = [None, 'b', 'c']
    self.assertFalse(equal(list_1, list_2))