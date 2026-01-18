from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def remote_gotPage(self, page):
    self.pages.append(page)