import random
import time
import unittest
def test_imperfect_hitrate(self):
    size = 1000
    cache = self._makeOne(size / 2)
    for count in range(size):
        cache.put(count, 'item%s' % count)
    hits = 0
    misses = 0
    total_gets = 0
    for cache_op in range(10000):
        item = random.randrange(0, size - 1)
        if random.getrandbits(1):
            entry = cache.get(item)
            total_gets += 1
            self.assertTrue(entry == 'item%s' % item or entry is None)
            if entry is None:
                misses += 1
            else:
                hits += 1
        else:
            cache.put(item, 'item%s' % item)
    hit_ratio = hits / float(total_gets) * 100
    self.assertTrue(hit_ratio > 45)
    self.assertTrue(hit_ratio < 55)
    internal_hit_ratio = 100 * cache.hits / cache.lookups
    self.assertTrue(internal_hit_ratio > 45)
    self.assertTrue(internal_hit_ratio < 55)
    internal_miss_ratio = 100 * cache.misses / cache.lookups
    self.assertTrue(internal_miss_ratio > 45)
    self.assertTrue(internal_miss_ratio < 55)
    self.check_cache_is_consistent(cache)