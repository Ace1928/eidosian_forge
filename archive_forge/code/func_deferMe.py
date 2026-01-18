from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def deferMe():
    d = Deferred()
    d.addErrback(deferMeMore)
    return d