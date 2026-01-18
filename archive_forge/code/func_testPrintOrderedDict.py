from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testPrintOrderedDict(self):
    with self.assertOutputMatches(stdout='A:\\s+A\\s+2:\\s+2\\s+', stderr=None):
        core.Fire(tc.OrderedDictionary, command=['non_empty'])
    with self.assertOutputMatches(stdout='{}'):
        core.Fire(tc.OrderedDictionary, command=['empty'])