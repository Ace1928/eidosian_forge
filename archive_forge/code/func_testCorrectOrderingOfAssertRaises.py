from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from fire import testutils
import six
def testCorrectOrderingOfAssertRaises(self):
    with self.assertOutputMatches(stdout='Yep.*first.*second'):
        with self.assertRaises(ValueError):
            print('Yep, this is the first line.\nThis is the second.')
            raise ValueError()