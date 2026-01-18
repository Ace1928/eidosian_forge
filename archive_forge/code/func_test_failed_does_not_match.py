from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_failed_does_not_match(self):
    fail = make_failure(RuntimeError('arbitrary failure'))
    deferred = defer.fail(fail)
    self.addCleanup(deferred.addErrback, lambda _: None)
    mismatch = self.match(deferred)
    self.assertThat(mismatch, mismatches(Equals('No result expected on %r, found %r instead' % (deferred, fail))))