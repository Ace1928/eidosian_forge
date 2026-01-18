import unittest
import cachetools.keys
def test_hashkey(self, key=cachetools.keys.hashkey):
    self.assertEqual(key(), key())
    self.assertEqual(hash(key()), hash(key()))
    self.assertEqual(key(1, 2, 3), key(1, 2, 3))
    self.assertEqual(hash(key(1, 2, 3)), hash(key(1, 2, 3)))
    self.assertEqual(key(1, 2, 3, x=0), key(1, 2, 3, x=0))
    self.assertEqual(hash(key(1, 2, 3, x=0)), hash(key(1, 2, 3, x=0)))
    self.assertNotEqual(key(1, 2, 3), key(3, 2, 1))
    self.assertNotEqual(key(1, 2, 3), key(1, 2, 3, x=None))
    self.assertNotEqual(key(1, 2, 3, x=0), key(1, 2, 3, x=None))
    self.assertNotEqual(key(1, 2, 3, x=0), key(1, 2, 3, y=0))
    with self.assertRaises(TypeError):
        hash(key({}))
    self.assertEqual(key(1, 2, 3), key(1.0, 2.0, 3.0))
    self.assertEqual(hash(key(1, 2, 3)), hash(key(1.0, 2.0, 3.0)))