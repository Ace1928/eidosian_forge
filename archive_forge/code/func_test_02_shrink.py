from __future__ import with_statement, division
from functools import partial
from passlib.utils import getrandbytes
from passlib.tests.utils import TestCase
def test_02_shrink(self):
    """shrink_des_key()"""
    from passlib.crypto.des import expand_des_key, shrink_des_key, INT_64_MASK
    rng = self.getRandom()
    for i in range(20):
        key1 = getrandbytes(rng, 7)
        key2 = expand_des_key(key1)
        key3 = shrink_des_key(key2)
        self.assertEqual(key3, key1)
    self.assertRaises(TypeError, shrink_des_key, 1.0)
    self.assertRaises(ValueError, shrink_des_key, INT_64_MASK + 1)
    self.assertRaises(ValueError, shrink_des_key, b'\x00' * 9)
    self.assertRaises(ValueError, shrink_des_key, -1)
    self.assertRaises(ValueError, shrink_des_key, b'\x00' * 7)