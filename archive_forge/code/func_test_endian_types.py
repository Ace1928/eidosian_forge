import unittest
from ctypes import *
import re, sys
def test_endian_types(self):
    for tp, fmt, shape, itemtp in endian_types:
        ob = tp()
        v = memoryview(ob)
        try:
            self.assertEqual(v.format, fmt)
            if shape:
                self.assertEqual(len(v), shape[0])
            else:
                self.assertEqual(len(v) * sizeof(itemtp), sizeof(ob))
            self.assertEqual(v.itemsize, sizeof(itemtp))
            self.assertEqual(v.shape, shape)
            self.assertFalse(v.readonly)
            if v.shape:
                n = 1
                for dim in v.shape:
                    n = n * dim
                self.assertEqual(n, len(v))
        except:
            print(tp)
            raise