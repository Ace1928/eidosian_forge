import unittest
def test_missing_getsizeof(self):

    class DefaultCache(self.Cache):

        def __missing__(self, key):
            try:
                self[key] = key
            except ValueError:
                pass
            return key
    cache = DefaultCache(maxsize=2, getsizeof=lambda x: x)
    self.assertEqual(0, cache.currsize)
    self.assertEqual(2, cache.maxsize)
    self.assertEqual(1, cache[1])
    self.assertEqual(1, len(cache))
    self.assertEqual(1, cache.currsize)
    self.assertIn(1, cache)
    self.assertEqual(2, cache[2])
    self.assertEqual(1, len(cache))
    self.assertEqual(2, cache.currsize)
    self.assertNotIn(1, cache)
    self.assertIn(2, cache)
    self.assertEqual(3, cache[3])
    self.assertEqual(1, len(cache))
    self.assertEqual(2, cache.currsize)
    self.assertEqual(1, cache[1])
    self.assertEqual(1, len(cache))
    self.assertEqual(1, cache.currsize)
    self.assertEqual((1, 1), cache.popitem())