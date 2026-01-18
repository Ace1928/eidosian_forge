import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_insertion_small(self):
    d = Dict(self, 4, 8)
    self.assertEqual(len(d), 0)
    self.assertIsNone(d.get('abcd'))
    d['abcd'] = 'beefcafe'
    self.assertEqual(len(d), 1)
    self.assertIsNotNone(d.get('abcd'))
    self.assertEqual(d['abcd'], 'beefcafe')
    d['abcd'] = 'cafe0000'
    self.assertEqual(len(d), 1)
    self.assertEqual(d['abcd'], 'cafe0000')
    d['abce'] = 'cafe0001'
    self.assertEqual(len(d), 2)
    self.assertEqual(d['abcd'], 'cafe0000')
    self.assertEqual(d['abce'], 'cafe0001')
    d['abcf'] = 'cafe0002'
    self.assertEqual(len(d), 3)
    self.assertEqual(d['abcd'], 'cafe0000')
    self.assertEqual(d['abce'], 'cafe0001')
    self.assertEqual(d['abcf'], 'cafe0002')