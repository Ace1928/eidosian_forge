from unittest import TestCase
import simplejson as json
def test_circular_list(self):
    lst = []
    lst.append(lst)
    self.assertRaises(ValueError, json.dumps, lst)