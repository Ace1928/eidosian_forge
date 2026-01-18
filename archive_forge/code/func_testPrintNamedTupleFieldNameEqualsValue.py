from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testPrintNamedTupleFieldNameEqualsValue(self):
    with self.assertOutputMatches(stdout='x', stderr=None):
        core.Fire(tc.NamedTuple, command=['matching_names', 'x'])