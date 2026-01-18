import unittest
import simplejson as json
def test_non_ascii_basic_encode(self):
    self.assertEqual('"\\u0026"', self.non_ascii_encoder.encode('&'))
    self.assertEqual('"\\u003c"', self.non_ascii_encoder.encode('<'))
    self.assertEqual('"\\u003e"', self.non_ascii_encoder.encode('>'))
    self.assertEqual('"\\u2028"', self.non_ascii_encoder.encode(u'\u2028'))