import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_message_chunks(self):
    auth_data = get_tag_random('authenticated data', 127)
    plaintext = get_tag_random('plaintext', 127)
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher.update(auth_data)
    ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

    def break_up(data, chunk_length):
        return [data[i:i + chunk_length] for i in range(0, len(data), chunk_length)]
    for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        for chunk in break_up(auth_data, chunk_length):
            cipher.update(chunk)
        pt2 = b('')
        for chunk in break_up(ciphertext, chunk_length):
            pt2 += cipher.decrypt(chunk)
        pt2 += cipher.decrypt()
        self.assertEqual(plaintext, pt2)
        cipher.verify(ref_mac)
    for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        for chunk in break_up(auth_data, chunk_length):
            cipher.update(chunk)
        ct2 = b('')
        for chunk in break_up(plaintext, chunk_length):
            ct2 += cipher.encrypt(chunk)
        ct2 += cipher.encrypt()
        self.assertEqual(ciphertext, ct2)
        self.assertEqual(cipher.digest(), ref_mac)