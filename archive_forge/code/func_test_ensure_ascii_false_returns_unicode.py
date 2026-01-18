import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_ensure_ascii_false_returns_unicode(self):
    self.assertEqual(type(json.dumps([], ensure_ascii=False)), text_type)
    self.assertEqual(type(json.dumps(0, ensure_ascii=False)), text_type)
    self.assertEqual(type(json.dumps({}, ensure_ascii=False)), text_type)
    self.assertEqual(type(json.dumps('', ensure_ascii=False)), text_type)