import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_get_user_encoding(self):
    self._encoding = 'user_encoding'
    fake_codec.add('user_encoding')
    self.assertEqual('iso8859-1', osutils.get_user_encoding())
    self.assertEqual('', sys.stderr.getvalue())