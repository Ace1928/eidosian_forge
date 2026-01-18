import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_encoding6(self):
    u = u'αΩ'
    j = json.dumps([u], ensure_ascii=False)
    self.assertEqual(j, u'["' + u + u'"]')