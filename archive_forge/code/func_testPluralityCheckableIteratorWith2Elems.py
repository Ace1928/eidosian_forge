from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
from six.moves import range
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
import gslib.tests.testcase as testcase
def testPluralityCheckableIteratorWith2Elems(self):
    """Tests PluralityCheckableIterator with 2 elements."""
    input_list = list(range(2))
    it = iter(input_list)
    pcit = PluralityCheckableIterator(it)
    self.assertFalse(pcit.IsEmpty())
    self.assertTrue(pcit.HasPlurality())
    output_list = list(pcit)
    self.assertEqual(input_list, output_list)