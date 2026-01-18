from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_different_succeeded_result_fails(self):
    result = object()
    deferred = defer.succeed(result)
    matcher = Is(None)
    mismatch = matcher.match(result)
    self.assertThat(self.match(matcher, deferred), mismatches(Equals(mismatch.describe()), Equals(mismatch.get_details())))