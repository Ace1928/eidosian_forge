from unittest import TestCase
import simplejson as json
def test_circular_off_default(self):
    json.dumps([set()], default=default_iterable, check_circular=False)
    self.assertRaises(TypeError, json.dumps, [set()], check_circular=False)