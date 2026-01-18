from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueStrings(self):
    self.assertEqual(parser.DefaultParseValue('hello'), 'hello')
    self.assertEqual(parser.DefaultParseValue('path/file.jpg'), 'path/file.jpg')
    self.assertEqual(parser.DefaultParseValue('hello world'), 'hello world')
    self.assertEqual(parser.DefaultParseValue('--flag'), '--flag')