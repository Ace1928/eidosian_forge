import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_ensure_ascii_false_bytestring_encoding(self):
    doc1 = {u'quux': b('ArrÃªt sur images')}
    doc2 = {u'quux': u'Arrêt sur images'}
    doc_ascii = '{"quux": "Arr\\u00eat sur images"}'
    doc_unicode = u'{"quux": "Arrêt sur images"}'
    self.assertEqual(json.dumps(doc1), doc_ascii)
    self.assertEqual(json.dumps(doc2), doc_ascii)
    self.assertEqual(json.dumps(doc1, ensure_ascii=False), doc_unicode)
    self.assertEqual(json.dumps(doc2, ensure_ascii=False), doc_unicode)