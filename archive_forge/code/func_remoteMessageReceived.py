import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def remoteMessageReceived(self, broker, message, args, kw):
    """A remote message has been received.  Dispatch it appropriately.

        The default implementation is to dispatch to a method called
        'C{observe_messagename}' and call it on my  with the same arguments.
        """
    if not isinstance(message, str):
        message = message.decode('utf8')
    args = broker.unserialize(args)
    kw = broker.unserialize(kw)
    method = getattr(self, 'observe_%s' % message)
    try:
        state = method(*args, **kw)
    except TypeError:
        log.msg(f"{method} didn't accept {args} and {kw}")
        raise
    return broker.serialize(state, None, method, args, kw)