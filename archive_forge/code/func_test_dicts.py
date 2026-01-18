from unittest import TestCase
import simplejson as json
def test_dicts(self):
    for val, expect in self.values:
        val = {'k': val}
        expect = {'k': expect}
        self.assertEqual(val, json.loads(json.dumps(val)))
        self.assertEqual(expect, json.loads(json.dumps(val, int_as_string_bitcount=31)))