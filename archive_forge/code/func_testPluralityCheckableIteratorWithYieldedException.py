from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
from six.moves import range
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
import gslib.tests.testcase as testcase
def testPluralityCheckableIteratorWithYieldedException(self):
    """Tests PCI with an iterator that yields an exception.

    The yielded exception is in the form of a tuple and must also contain a
    stack trace.
    """

    class IterTest(six.Iterator):

        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.position == 0:
                try:
                    self.position += 1
                    raise CustomTestException('Test exception 0')
                except CustomTestException as e:
                    return (e, sys.exc_info()[2])
            elif self.position == 1:
                self.position += 1
                return 1
            else:
                raise StopIteration()
    pcit = PluralityCheckableIterator(IterTest())
    iterated_value = None
    try:
        for _ in pcit:
            pass
        self.fail('Expected exception 0 from iterator')
    except CustomTestException as e:
        self.assertIn(str(e), 'Test exception 0')
    for value in pcit:
        iterated_value = value
    self.assertEqual(iterated_value, 1)