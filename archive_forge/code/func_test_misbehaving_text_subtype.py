from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_misbehaving_text_subtype(self):
    text = 'this is some text'
    self.assertEqual(json.dumps(MisbehavingTextSubtype(text)), json.dumps(text))
    self.assertEqual(json.dumps([MisbehavingTextSubtype(text)]), json.dumps([text]))
    self.assertEqual(json.dumps({MisbehavingTextSubtype(text): 42}), json.dumps({text: 42}))