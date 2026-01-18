from __future__ import with_statement, division
from functools import partial
from passlib.utils import getrandbytes
from passlib.tests.utils import TestCase
def test_03_encrypt_bytes(self):
    """des_encrypt_block()"""
    from passlib.crypto.des import des_encrypt_block, shrink_des_key, _pack64, _unpack64
    for key, plaintext, correct in self.des_test_vectors:
        key = _pack64(key)
        plaintext = _pack64(plaintext)
        correct = _pack64(correct)
        result = des_encrypt_block(key, plaintext)
        self.assertEqual(result, correct, 'key=%r plaintext=%r:' % (key, plaintext))
        key2 = shrink_des_key(key)
        result = des_encrypt_block(key2, plaintext)
        self.assertEqual(result, correct, 'key=%r shrink(key)=%r plaintext=%r:' % (key, key2, plaintext))
        for _ in range(20):
            key3 = _pack64(self._random_parity(_unpack64(key)))
            result = des_encrypt_block(key3, plaintext)
            self.assertEqual(result, correct, 'key=%r rndparity(key)=%r plaintext=%r:' % (key, key3, plaintext))
    stub = b'\x00' * 8
    self.assertRaises(TypeError, des_encrypt_block, 0, stub)
    self.assertRaises(ValueError, des_encrypt_block, b'\x00' * 6, stub)
    self.assertRaises(TypeError, des_encrypt_block, stub, 0)
    self.assertRaises(ValueError, des_encrypt_block, stub, b'\x00' * 7)
    self.assertRaises(ValueError, des_encrypt_block, stub, stub, salt=-1)
    self.assertRaises(ValueError, des_encrypt_block, stub, stub, salt=1 << 24)
    self.assertRaises(ValueError, des_encrypt_block, stub, stub, 0, rounds=0)