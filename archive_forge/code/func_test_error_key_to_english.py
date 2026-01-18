import unittest
import binascii
from Cryptodome.Util.RFC1751 import key_to_english, english_to_key
def test_error_key_to_english(self):
    self.assertRaises(ValueError, key_to_english, b'0' * 7)