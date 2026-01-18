from unittest import TestCase
import simplejson as json
def test_ints_outside_range_fails(self):
    self.assertNotEqual(str(1 << 15), json.loads(json.dumps(1 << 15, int_as_string_bitcount=16)))