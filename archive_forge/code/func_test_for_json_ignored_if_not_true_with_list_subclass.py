import unittest
import simplejson as json
def test_for_json_ignored_if_not_true_with_list_subclass(self):
    for for_json in (None, False):
        self.assertRoundTrip(ListForJson(['l']), ['l'], for_json=for_json)