from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from fire import testutils
import six
def testNoCheckOnException(self):
    with self.assertRaises(ValueError):
        with self.assertOutputMatches(stdout='blah'):
            raise ValueError()