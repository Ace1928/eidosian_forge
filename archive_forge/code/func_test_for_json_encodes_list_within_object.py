import unittest
import simplejson as json
def test_for_json_encodes_list_within_object(self):
    self.assertRoundTrip({'nested': ForJsonList()}, {'nested': ForJsonList().for_json()})