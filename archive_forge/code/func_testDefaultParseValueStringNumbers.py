from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueStringNumbers(self):
    self.assertEqual(parser.DefaultParseValue("'23'"), '23')
    self.assertEqual(parser.DefaultParseValue("'23.0'"), '23.0')
    self.assertEqual(parser.DefaultParseValue("'23.5'"), '23.5')
    self.assertEqual(parser.DefaultParseValue('"23"'), '23')
    self.assertEqual(parser.DefaultParseValue('"23.0"'), '23.0')
    self.assertEqual(parser.DefaultParseValue('"23.5"'), '23.5')