from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueQuotedStringNumbers(self):
    self.assertEqual(parser.DefaultParseValue('"\'123\'"'), "'123'")