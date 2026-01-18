import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_90_checksums(self):
    """test internal parsing of 'checksum' keyword"""
    self.assertRaises(TypeError, self.handler, use_defaults=True, checksum={'sha-1': u('X') * 20})
    self.assertRaises(ValueError, self.handler, use_defaults=True, checksum={'sha-256': b'X' * 32})