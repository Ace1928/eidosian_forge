from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueBadLiteral(self):
    self.assertEqual(parser.DefaultParseValue('[(A, 2, "3"), 5'), '[(A, 2, "3"), 5')
    self.assertEqual(parser.DefaultParseValue('x=10'), 'x=10')