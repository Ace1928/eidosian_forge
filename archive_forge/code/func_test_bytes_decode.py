from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_bytes_decode(self):
    cls = json.decoder.JSONDecoder
    data = b('"â\x82¬"')
    self.assertEqual(cls().decode(data), u'€')
    self.assertEqual(cls(encoding='latin1').decode(data), u'â\x82¬')
    self.assertEqual(cls(encoding=None).decode(data), u'€')
    data = MisbehavingBytesSubtype(b('"â\x82¬"'))
    self.assertEqual(cls().decode(data), u'€')
    self.assertEqual(cls(encoding='latin1').decode(data), u'â\x82¬')
    self.assertEqual(cls(encoding=None).decode(data), u'€')