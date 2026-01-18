import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_default_encoding(self):
    self.assertEqual(json.loads(u'{"a": "é"}'.encode('utf-8')), {'a': u'é'})