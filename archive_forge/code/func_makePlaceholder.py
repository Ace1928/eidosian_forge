from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def makePlaceholder(deferred, name):

    def placeholder(*args, **kw):
        deferred.addCallback(lambda r: getattr(r, name)(*args, **kw))
        return deferred
    return placeholder