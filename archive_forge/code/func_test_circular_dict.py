from unittest import TestCase
import simplejson as json
def test_circular_dict(self):
    dct = {}
    dct['a'] = dct
    self.assertRaises(ValueError, json.dumps, dct)