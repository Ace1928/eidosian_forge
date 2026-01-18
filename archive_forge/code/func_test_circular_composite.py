from unittest import TestCase
import simplejson as json
def test_circular_composite(self):
    dct2 = {}
    dct2['a'] = []
    dct2['a'].append(dct2)
    self.assertRaises(ValueError, json.dumps, dct2)