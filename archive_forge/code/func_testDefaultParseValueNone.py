from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueNone(self):
    self.assertEqual(parser.DefaultParseValue('None'), None)