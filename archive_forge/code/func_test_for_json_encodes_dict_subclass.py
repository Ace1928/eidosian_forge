import unittest
import simplejson as json
def test_for_json_encodes_dict_subclass(self):
    self.assertRoundTrip(DictForJson(a=1), DictForJson(a=1).for_json())