from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def getAllPages(referenceable, methodName, *args, **kw):
    """
    A utility method that will call a remote method which expects a
    PageCollector as the first argument.
    """
    d = defer.Deferred()
    referenceable.callRemote(methodName, CallbackPageCollector(d.callback), *args, **kw)
    return d