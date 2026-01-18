from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def test_lookup_missing_key_before_all_others(self):
    calls = []

    def missing_first_content(location_keys):
        calls.append(location_keys)
        result = []
        for location_key in location_keys:
            if location_key[0] == 0:
                result.append((location_key, False))
            else:
                result.append((location_key, -1))
        return result
    self.assertEqual([], list(bisect_multi_bytes(missing_first_content, 0, ['foo', 'bar'])))
    self.assertEqual([[(0, 'foo'), (0, 'bar')]], calls)
    del calls[:]
    self.assertEqual([], list(bisect_multi_bytes(missing_first_content, 2, ['foo', 'bar'])))
    self.assertEqual([[(1, 'foo'), (1, 'bar')], [(0, 'foo'), (0, 'bar')]], calls)
    del calls[:]
    self.assertEqual([], list(bisect_multi_bytes(missing_first_content, 268435456 - 1, ['foo', 'bar'])))
    self.assertEqual([[(134217727, 'foo'), (134217727, 'bar')], [(67108864, 'foo'), (67108864, 'bar')], [(33554433, 'foo'), (33554433, 'bar')], [(16777218, 'foo'), (16777218, 'bar')], [(8388611, 'foo'), (8388611, 'bar')], [(4194308, 'foo'), (4194308, 'bar')], [(2097157, 'foo'), (2097157, 'bar')], [(1048582, 'foo'), (1048582, 'bar')], [(524295, 'foo'), (524295, 'bar')], [(262152, 'foo'), (262152, 'bar')], [(131081, 'foo'), (131081, 'bar')], [(65546, 'foo'), (65546, 'bar')], [(32779, 'foo'), (32779, 'bar')], [(16396, 'foo'), (16396, 'bar')], [(8205, 'foo'), (8205, 'bar')], [(4110, 'foo'), (4110, 'bar')], [(2063, 'foo'), (2063, 'bar')], [(1040, 'foo'), (1040, 'bar')], [(529, 'foo'), (529, 'bar')], [(274, 'foo'), (274, 'bar')], [(147, 'foo'), (147, 'bar')], [(84, 'foo'), (84, 'bar')], [(53, 'foo'), (53, 'bar')], [(38, 'foo'), (38, 'bar')], [(31, 'foo'), (31, 'bar')], [(28, 'foo'), (28, 'bar')], [(27, 'foo'), (27, 'bar')], [(26, 'foo'), (26, 'bar')], [(25, 'foo'), (25, 'bar')], [(24, 'foo'), (24, 'bar')], [(23, 'foo'), (23, 'bar')], [(22, 'foo'), (22, 'bar')], [(21, 'foo'), (21, 'bar')], [(20, 'foo'), (20, 'bar')], [(19, 'foo'), (19, 'bar')], [(18, 'foo'), (18, 'bar')], [(17, 'foo'), (17, 'bar')], [(16, 'foo'), (16, 'bar')], [(15, 'foo'), (15, 'bar')], [(14, 'foo'), (14, 'bar')], [(13, 'foo'), (13, 'bar')], [(12, 'foo'), (12, 'bar')], [(11, 'foo'), (11, 'bar')], [(10, 'foo'), (10, 'bar')], [(9, 'foo'), (9, 'bar')], [(8, 'foo'), (8, 'bar')], [(7, 'foo'), (7, 'bar')], [(6, 'foo'), (6, 'bar')], [(5, 'foo'), (5, 'bar')], [(4, 'foo'), (4, 'bar')], [(3, 'foo'), (3, 'bar')], [(2, 'foo'), (2, 'bar')], [(1, 'foo'), (1, 'bar')], [(0, 'foo'), (0, 'bar')]], calls)