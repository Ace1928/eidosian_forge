from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def test_lookup_missing_key_no_content(self):
    """Doing a lookup in a zero-length file still does a single request.

        This makes sense because the bisector cannot tell how long content is
        and its more flexible to only stop when the content object says 'False'
        for a given location, key pair.
        """
    calls = []

    def missing_content(location_keys):
        calls.append(location_keys)
        return ((location_key, False) for location_key in location_keys)
    self.assertEqual([], list(bisect_multi_bytes(missing_content, 0, ['foo', 'bar'])))
    self.assertEqual([[(0, 'foo'), (0, 'bar')]], calls)