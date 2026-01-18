from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueBareWordsLists(self):
    self.assertEqual(parser.DefaultParseValue('[one, 2, "3"]'), ['one', 2, '3'])