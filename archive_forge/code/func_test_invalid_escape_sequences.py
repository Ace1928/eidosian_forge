import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_invalid_escape_sequences(self):
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u1')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u12')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u123')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u1234')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u123x"')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u12x4"')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\u1x34"')
    self.assertRaises(json.JSONDecodeError, json.loads, '"\\ux234"')
    if sys.maxunicode > 65535:
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u0"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u00"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u000"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u000x"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u00x0"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\u0x00"')
        self.assertRaises(json.JSONDecodeError, json.loads, '"\\ud800\\ux000"')