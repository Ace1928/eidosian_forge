from __future__ import with_statement, division
from functools import partial
from passlib.utils import getrandbytes
from passlib.tests.utils import TestCase
def test_04_encrypt_ints(self):
    """des_encrypt_int_block()"""
    from passlib.crypto.des import des_encrypt_int_block
    for key, plaintext, correct in self.des_test_vectors:
        result = des_encrypt_int_block(key, plaintext)
        self.assertEqual(result, correct, 'key=%r plaintext=%r:' % (key, plaintext))
        for _ in range(20):
            key3 = self._random_parity(key)
            result = des_encrypt_int_block(key3, plaintext)
            self.assertEqual(result, correct, 'key=%r rndparity(key)=%r plaintext=%r:' % (key, key3, plaintext))
    self.assertRaises(TypeError, des_encrypt_int_block, b'\x00', 0)
    self.assertRaises(ValueError, des_encrypt_int_block, -1, 0)
    self.assertRaises(TypeError, des_encrypt_int_block, 0, b'\x00')
    self.assertRaises(ValueError, des_encrypt_int_block, 0, -1)
    self.assertRaises(ValueError, des_encrypt_int_block, 0, 0, salt=-1)
    self.assertRaises(ValueError, des_encrypt_int_block, 0, 0, salt=1 << 24)
    self.assertRaises(ValueError, des_encrypt_int_block, 0, 0, 0, rounds=0)