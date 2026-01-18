import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_encoding4(self):
    u = u'αΩ'
    j = json.dumps([u])
    self.assertEqual(j, '["\\u03b1\\u03a9"]')