from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueIgnoreBinOp(self):
    self.assertEqual(parser.DefaultParseValue('2017-10-10'), '2017-10-10')
    self.assertEqual(parser.DefaultParseValue('1+1'), '1+1')