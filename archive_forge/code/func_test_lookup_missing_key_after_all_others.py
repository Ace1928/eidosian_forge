from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def test_lookup_missing_key_after_all_others(self):
    calls = []
    end = None

    def missing_last_content(location_keys):
        calls.append(location_keys)
        result = []
        for location_key in location_keys:
            if location_key[0] == end:
                result.append((location_key, False))
            else:
                result.append((location_key, +1))
        return result
    end = 0
    self.assertEqual([], list(bisect_multi_bytes(missing_last_content, 0, ['foo', 'bar'])))
    self.assertEqual([[(0, 'foo'), (0, 'bar')]], calls)
    del calls[:]
    end = 2
    self.assertEqual([], list(bisect_multi_bytes(missing_last_content, 3, ['foo', 'bar'])))
    self.assertEqual([[(1, 'foo'), (1, 'bar')], [(2, 'foo'), (2, 'bar')]], calls)
    del calls[:]
    end = 268435456 - 2
    self.assertEqual([], list(bisect_multi_bytes(missing_last_content, 268435456 - 1, ['foo', 'bar'])))
    self.assertEqual([[(134217727, 'foo'), (134217727, 'bar')], [(201326590, 'foo'), (201326590, 'bar')], [(234881021, 'foo'), (234881021, 'bar')], [(251658236, 'foo'), (251658236, 'bar')], [(260046843, 'foo'), (260046843, 'bar')], [(264241146, 'foo'), (264241146, 'bar')], [(266338297, 'foo'), (266338297, 'bar')], [(267386872, 'foo'), (267386872, 'bar')], [(267911159, 'foo'), (267911159, 'bar')], [(268173302, 'foo'), (268173302, 'bar')], [(268304373, 'foo'), (268304373, 'bar')], [(268369908, 'foo'), (268369908, 'bar')], [(268402675, 'foo'), (268402675, 'bar')], [(268419058, 'foo'), (268419058, 'bar')], [(268427249, 'foo'), (268427249, 'bar')], [(268431344, 'foo'), (268431344, 'bar')], [(268433391, 'foo'), (268433391, 'bar')], [(268434414, 'foo'), (268434414, 'bar')], [(268434925, 'foo'), (268434925, 'bar')], [(268435180, 'foo'), (268435180, 'bar')], [(268435307, 'foo'), (268435307, 'bar')], [(268435370, 'foo'), (268435370, 'bar')], [(268435401, 'foo'), (268435401, 'bar')], [(268435416, 'foo'), (268435416, 'bar')], [(268435423, 'foo'), (268435423, 'bar')], [(268435426, 'foo'), (268435426, 'bar')], [(268435427, 'foo'), (268435427, 'bar')], [(268435428, 'foo'), (268435428, 'bar')], [(268435429, 'foo'), (268435429, 'bar')], [(268435430, 'foo'), (268435430, 'bar')], [(268435431, 'foo'), (268435431, 'bar')], [(268435432, 'foo'), (268435432, 'bar')], [(268435433, 'foo'), (268435433, 'bar')], [(268435434, 'foo'), (268435434, 'bar')], [(268435435, 'foo'), (268435435, 'bar')], [(268435436, 'foo'), (268435436, 'bar')], [(268435437, 'foo'), (268435437, 'bar')], [(268435438, 'foo'), (268435438, 'bar')], [(268435439, 'foo'), (268435439, 'bar')], [(268435440, 'foo'), (268435440, 'bar')], [(268435441, 'foo'), (268435441, 'bar')], [(268435442, 'foo'), (268435442, 'bar')], [(268435443, 'foo'), (268435443, 'bar')], [(268435444, 'foo'), (268435444, 'bar')], [(268435445, 'foo'), (268435445, 'bar')], [(268435446, 'foo'), (268435446, 'bar')], [(268435447, 'foo'), (268435447, 'bar')], [(268435448, 'foo'), (268435448, 'bar')], [(268435449, 'foo'), (268435449, 'bar')], [(268435450, 'foo'), (268435450, 'bar')], [(268435451, 'foo'), (268435451, 'bar')], [(268435452, 'foo'), (268435452, 'bar')], [(268435453, 'foo'), (268435453, 'bar')], [(268435454, 'foo'), (268435454, 'bar')]], calls)