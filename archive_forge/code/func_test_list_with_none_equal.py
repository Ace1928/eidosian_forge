from unittest import TestCase
from jsonschema._utils import equal
def test_list_with_none_equal(self):
    list_1 = ['a', None, 'c']
    list_2 = ['a', None, 'c']
    self.assertTrue(equal(list_1, list_2))