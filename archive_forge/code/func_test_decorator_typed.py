import unittest
import cachetools
import cachetools.keys
def test_decorator_typed(self):
    cache = self.cache(3)
    key = cachetools.keys.typedkey
    wrapper = cachetools.cached(cache, key=key)(self.func)
    self.assertEqual(len(cache), 0)
    self.assertEqual(wrapper.__wrapped__, self.func)
    self.assertEqual(wrapper(0), 0)
    self.assertEqual(len(cache), 1)
    self.assertIn(cachetools.keys.typedkey(0), cache)
    self.assertNotIn(cachetools.keys.typedkey(1), cache)
    self.assertNotIn(cachetools.keys.typedkey(1.0), cache)
    self.assertEqual(wrapper(1), 1)
    self.assertEqual(len(cache), 2)
    self.assertIn(cachetools.keys.typedkey(0), cache)
    self.assertIn(cachetools.keys.typedkey(1), cache)
    self.assertNotIn(cachetools.keys.typedkey(1.0), cache)
    self.assertEqual(wrapper(1), 1)
    self.assertEqual(len(cache), 2)
    self.assertEqual(wrapper(1.0), 2)
    self.assertEqual(len(cache), 3)
    self.assertIn(cachetools.keys.typedkey(0), cache)
    self.assertIn(cachetools.keys.typedkey(1), cache)
    self.assertIn(cachetools.keys.typedkey(1.0), cache)
    self.assertEqual(wrapper(1.0), 2)
    self.assertEqual(len(cache), 3)