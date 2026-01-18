from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def test_lookup_no_keys_no_calls(self):
    calls = []

    def missing_content(location_keys):
        calls.append(location_keys)
        return ((location_key, False) for location_key in location_keys)
    self.assertEqual([], list(bisect_multi_bytes(missing_content, 100, [])))
    self.assertEqual([], calls)