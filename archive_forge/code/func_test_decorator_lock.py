import unittest
import cachetools
import cachetools.keys
def test_decorator_lock(self):

    class Lock:
        count = 0

        def __enter__(self):
            Lock.count += 1

        def __exit__(self, *exc):
            pass
    cache = self.cache(2)
    wrapper = cachetools.cached(cache, lock=Lock())(self.func)
    self.assertEqual(len(cache), 0)
    self.assertEqual(wrapper.__wrapped__, self.func)
    self.assertEqual(wrapper(0), 0)
    self.assertEqual(Lock.count, 2)
    self.assertEqual(wrapper(1), 1)
    self.assertEqual(Lock.count, 4)
    self.assertEqual(wrapper(1), 1)
    self.assertEqual(Lock.count, 5)