from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
from six.moves import range
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
import gslib.tests.testcase as testcase
def testPluralityCheckableIteratorWith1Elem1Exception(self):
    """Tests PluralityCheckableIterator with 2 elements.

    The second element raises an exception.
    """

    class IterTest(six.Iterator):

        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.position == 0:
                self.position += 1
                return 1
            elif self.position == 1:
                self.position += 1
                raise CustomTestException('Test exception')
            else:
                raise StopIteration()
    pcit = PluralityCheckableIterator(IterTest())
    self.assertFalse(pcit.IsEmpty())
    self.assertTrue(pcit.HasPlurality())
    iterated_value = None
    try:
        for value in pcit:
            iterated_value = value
        self.fail('Expected exception from iterator')
    except CustomTestException:
        pass
    self.assertEqual(iterated_value, 1)