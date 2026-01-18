import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
def identityPumpPolicy(queue, target):
    """
    L{identityPumpPolicy} is a policy which delivers each chunk of data written
    to the given queue as-is to the target.

    This isn't a particularly realistic policy.

    @see: L{loopbackAsync}
    """
    while queue:
        bytes = queue.get()
        if bytes is None:
            break
        target.dataReceived(bytes)