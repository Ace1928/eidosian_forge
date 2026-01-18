from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def get_mismatch_info(mismatch):
    return {'description': mismatch.describe(), 'details': mismatch.get_details()}