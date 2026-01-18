import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_mutability(self):
    l = List(self, 8, 1)
    one = struct.pack('q', 1)
    l.append(one)
    self.assertTrue(l.is_mutable)
    self.assertEqual(len(l), 1)
    r = struct.unpack('q', l[0])[0]
    self.assertEqual(r, 1)
    l.set_immutable()
    self.assertFalse(l.is_mutable)
    with self.assertRaises(ValueError) as raises:
        l.append(one)
    self.assertIn('list is immutable', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        l[0] = one
    self.assertIn('list is immutable', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        l.pop()
    self.assertIn('list is immutable', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        del l[0]
    self.assertIn('list is immutable', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        del l[0:1:1]
    self.assertIn('list is immutable', str(raises.exception))
    l.set_mutable()
    self.assertTrue(l.is_mutable)
    self.assertEqual(len(l), 1)
    r = struct.unpack('q', l[0])[0]
    self.assertEqual(r, 1)