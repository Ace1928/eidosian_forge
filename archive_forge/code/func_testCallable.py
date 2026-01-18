from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testCallable(self):
    with self.assertOutputMatches(stdout='foo:\\s+foo\\s+', stderr=None):
        core.Fire(tc.CallableWithKeywordArgument(), command=['--foo=foo'])
    with self.assertOutputMatches(stdout='foo\\s+', stderr=None):
        core.Fire(tc.CallableWithKeywordArgument(), command=['print_msg', 'foo'])
    with self.assertOutputMatches(stdout='', stderr=None):
        core.Fire(tc.CallableWithKeywordArgument(), command=[])