import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_big_unicode_decode(self):
    u = u'z𝄠x'
    self.assertEqual(json.loads('"' + u + '"'), u)
    self.assertEqual(json.loads('"z\\ud834\\udd20x"'), u)