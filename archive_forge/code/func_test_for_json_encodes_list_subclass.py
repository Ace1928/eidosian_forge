import unittest
import simplejson as json
def test_for_json_encodes_list_subclass(self):
    self.assertRoundTrip(ListForJson(['l']), ListForJson(['l']).for_json())