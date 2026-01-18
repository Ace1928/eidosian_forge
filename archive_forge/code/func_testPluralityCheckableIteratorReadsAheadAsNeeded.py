from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
from six.moves import range
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
import gslib.tests.testcase as testcase
def testPluralityCheckableIteratorReadsAheadAsNeeded(self):
    """Tests that the PCI does not unnecessarily read new elements."""

    class IterTest(six.Iterator):

        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.position == 3:
                raise StopIteration()
            self.position += 1
    pcit = PluralityCheckableIterator(IterTest())
    pcit.IsEmpty()
    pcit.PeekException()
    self.assertEqual(pcit.orig_iterator.position, 1)
    pcit.HasPlurality()
    self.assertEqual(pcit.orig_iterator.position, 2)
    next(pcit)
    self.assertEqual(pcit.orig_iterator.position, 2)
    next(pcit)
    self.assertEqual(pcit.orig_iterator.position, 2)
    next(pcit)
    self.assertEqual(pcit.orig_iterator.position, 3)
    try:
        next(pcit)
        self.fail('Expected StopIteration')
    except StopIteration:
        pass