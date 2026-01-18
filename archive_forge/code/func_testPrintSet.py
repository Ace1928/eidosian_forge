from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testPrintSet(self):
    with self.assertOutputMatches(stdout='.*three.*', stderr=None):
        core.Fire(tc.simple_set(), command=[])