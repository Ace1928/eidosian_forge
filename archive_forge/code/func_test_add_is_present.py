from .. import fifo_cache, tests
def test_add_is_present(self):
    c = fifo_cache.FIFOSizeCache()
    c[1] = '2'
    self.assertTrue(1 in c)
    self.assertEqual(1, len(c))
    self.assertEqual('2', c[1])
    self.assertEqual('2', c.get(1))
    self.assertEqual('2', c.get(1, None))
    self.assertEqual([1], list(c))
    self.assertEqual({1}, c.keys())
    self.assertEqual([(1, '2')], sorted(c.items()))
    self.assertEqual(['2'], sorted(c.values()))
    self.assertEqual({1: '2'}, c)
    self.assertEqual(1024 * 1024, c.cache_size())