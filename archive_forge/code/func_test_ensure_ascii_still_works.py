import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_ensure_ascii_still_works(self):
    for c in map(unichr, range(0, 127)):
        self.assertEqual(json.dumps(c, ensure_ascii=False), json.dumps(c))
    snowman = u'☃'
    self.assertEqual(json.dumps(c, ensure_ascii=False), '"' + c + '"')