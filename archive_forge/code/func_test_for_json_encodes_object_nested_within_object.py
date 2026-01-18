import unittest
import simplejson as json
def test_for_json_encodes_object_nested_within_object(self):
    self.assertRoundTrip(NestedForJson(), {'nested': {'for_json': 1}})