from .. import fifo_cache, tests
def test_copy_not_implemented(self):
    c = fifo_cache.FIFOCache()
    self.assertRaises(NotImplementedError, c.copy)