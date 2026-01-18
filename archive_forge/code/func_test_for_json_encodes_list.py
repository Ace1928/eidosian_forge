import unittest
import simplejson as json
def test_for_json_encodes_list(self):
    self.assertRoundTrip(ForJsonList(), ForJsonList().for_json())