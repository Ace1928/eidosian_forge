from unittest import TestCase
import simplejson as json
def test_circular_default(self):
    json.dumps([set()], default=default_iterable)
    self.assertRaises(TypeError, json.dumps, [set()])