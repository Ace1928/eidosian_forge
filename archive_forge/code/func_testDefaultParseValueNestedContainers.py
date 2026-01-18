from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueNestedContainers(self):
    self.assertEqual(parser.DefaultParseValue('[(A, 2, "3"), 5, {alpha: 10.2, beta: "cat"}]'), [('A', 2, '3'), 5, {'alpha': 10.2, 'beta': 'cat'}])