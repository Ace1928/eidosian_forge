from .. import fifo_cache, tests
def test_cleanup_funcs(self):
    log = []

    def logging_cleanup(key, value):
        log.append((key, value))
    c = fifo_cache.FIFOCache(5, 4)
    c.add(1, 2, cleanup=logging_cleanup)
    c.add(2, 3, cleanup=logging_cleanup)
    c.add(3, 4, cleanup=logging_cleanup)
    c.add(4, 5, cleanup=None)
    c[5] = 6
    self.assertEqual([], log)
    c.add(6, 7, cleanup=logging_cleanup)
    self.assertEqual([(1, 2), (2, 3)], log)
    del log[:]
    c.add(3, 8, cleanup=logging_cleanup)
    self.assertEqual([(3, 4)], log)
    del log[:]
    c[3] = 9
    self.assertEqual([(3, 8)], log)
    del log[:]
    c.clear()
    self.assertEqual([(6, 7)], log)
    del log[:]
    c.add(8, 9, cleanup=logging_cleanup)
    del c[8]
    self.assertEqual([(8, 9)], log)