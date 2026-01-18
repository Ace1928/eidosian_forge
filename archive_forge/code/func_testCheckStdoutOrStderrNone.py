from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from fire import testutils
import six
def testCheckStdoutOrStderrNone(self):
    with six.assertRaisesRegex(self, AssertionError, 'stdout:'):
        with self.assertOutputMatches(stdout=None):
            print('blah')
    with six.assertRaisesRegex(self, AssertionError, 'stderr:'):
        with self.assertOutputMatches(stderr=None):
            print('blah', file=sys.stderr)
    with six.assertRaisesRegex(self, AssertionError, 'stderr:'):
        with self.assertOutputMatches(stdout='apple', stderr=None):
            print('apple')
            print('blah', file=sys.stderr)