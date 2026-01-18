import random
import time
import unittest
def check_cache_is_consistent(self, cache):
    self.assertTrue(cache.hand < len(cache.clock_keys))
    self.assertTrue(cache.hand >= 0)
    self.assertEqual(cache.maxpos, cache.size - 1)
    self.assertEqual(len(cache.clock_keys), cache.size)
    self.assertEqual(len(cache.clock_keys), len(cache.clock_refs))
    self.assertTrue(len(cache.data) <= len(cache.clock_refs))
    for key, value in cache.data.items():
        pos, val, timeout = value
        self.assertTrue(type(pos) == type(42) or type(pos) == type(2 ** 128))
        self.assertTrue(pos >= 0)
        self.assertTrue(pos <= cache.maxpos)
        clock_key = cache.clock_keys[pos]
        self.assertTrue(clock_key is key)
        clock_ref = cache.clock_refs[pos]
        self.assertTrue(type(timeout) == type(3.141))
    for clock_ref in cache.clock_refs:
        self.assertTrue(clock_ref is True or clock_ref is False)