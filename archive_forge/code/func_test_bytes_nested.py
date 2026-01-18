from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_bytes_nested(self):
    self.assertEqual(json.dumps([b('â\x82¬')]), '["\\u20ac"]')
    self.assertRaises(UnicodeDecodeError, json.dumps, [b('¤')])
    self.assertEqual(json.dumps([b('¤')], encoding='iso-8859-1'), '["\\u00a4"]')
    self.assertEqual(json.dumps([b('¤')], encoding='iso-8859-15'), '["\\u20ac"]')
    if PY3:
        self.assertRaises(TypeError, json.dumps, [b('â\x82¬')], encoding=None)
        self.assertRaises(TypeError, json.dumps, [b('¤')], encoding=None)
        self.assertEqual(json.dumps([b('¤')], encoding=None, default=decode_iso_8859_15), '["\\u20ac"]')
    else:
        self.assertEqual(json.dumps([b('â\x82¬')], encoding=None), '["\\u20ac"]')
        self.assertRaises(UnicodeDecodeError, json.dumps, [b('¤')], encoding=None)
        self.assertRaises(UnicodeDecodeError, json.dumps, [b('¤')], encoding=None, default=decode_iso_8859_15)