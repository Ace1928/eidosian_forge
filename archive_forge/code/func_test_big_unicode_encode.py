import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_big_unicode_encode(self):
    u = u'𝄠'
    self.assertEqual(json.dumps(u), '"\\ud834\\udd20"')
    self.assertEqual(json.dumps(u, ensure_ascii=False), u'"𝄠"')