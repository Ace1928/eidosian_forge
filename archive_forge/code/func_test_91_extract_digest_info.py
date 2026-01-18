import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_91_extract_digest_info(self):
    """test scram.extract_digest_info()"""
    edi = self.handler.extract_digest_info
    h = '$scram$10$AAAAAA$sha-1=AQ,bbb=Ag,ccc=Aw'
    s = b'\x00' * 4
    self.assertEqual(edi(h, 'SHA1'), (s, 10, b'\x01'))
    self.assertEqual(edi(h, 'bbb'), (s, 10, b'\x02'))
    self.assertEqual(edi(h, 'ccc'), (s, 10, b'\x03'))
    self.assertRaises(KeyError, edi, h, 'ddd')
    c = '$scram$10$....$sha-1,bbb,ccc'
    self.assertRaises(ValueError, edi, c, 'sha-1')
    self.assertRaises(ValueError, edi, c, 'bbb')
    self.assertRaises(ValueError, edi, c, 'ddd')