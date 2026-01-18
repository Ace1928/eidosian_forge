from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def remote_endedPaging(self):
    self.callback(self.pages)