from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_bytes_key(self):
    self.assertEqual(json.dumps({b('â\x82¬'): 42}), '{"\\u20ac": 42}')
    self.assertRaises(UnicodeDecodeError, json.dumps, {b('¤'): 42})
    self.assertEqual(json.dumps({b('¤'): 42}, encoding='iso-8859-1'), '{"\\u00a4": 42}')
    self.assertEqual(json.dumps({b('¤'): 42}, encoding='iso-8859-15'), '{"\\u20ac": 42}')
    if PY3:
        self.assertRaises(TypeError, json.dumps, {b('â\x82¬'): 42}, encoding=None)
        self.assertRaises(TypeError, json.dumps, {b('¤'): 42}, encoding=None)
        self.assertRaises(TypeError, json.dumps, {b('¤'): 42}, encoding=None, default=decode_iso_8859_15)
        self.assertEqual(json.dumps({b('¤'): 42}, encoding=None, skipkeys=True), '{}')
    else:
        self.assertEqual(json.dumps({b('â\x82¬'): 42}, encoding=None), '{"\\u20ac": 42}')
        self.assertRaises(UnicodeDecodeError, json.dumps, {b('¤'): 42}, encoding=None)
        self.assertRaises(UnicodeDecodeError, json.dumps, {b('¤'): 42}, encoding=None, default=decode_iso_8859_15)
        self.assertRaises(UnicodeDecodeError, json.dumps, {b('¤'): 42}, encoding=None, skipkeys=True)