import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_fake_codec(self):
    self.assertRaises(LookupError, codecs.lookup, 'fake')
    fake_codec.add('fake')
    codecs.lookup('fake')