import errno
from select import (
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
def doPoll(self, timeout):
    """Poll the poller for new events."""
    if timeout is not None:
        timeout = int(timeout * 1000)
    try:
        l = self._poller.poll(timeout)
    except SelectError as e:
        if e.args[0] == errno.EINTR:
            return
        else:
            raise
    _drdw = self._doReadOrWrite
    for fd, event in l:
        try:
            selectable = self._selectables[fd]
        except KeyError:
            continue
        log.callWithLogger(selectable, _drdw, selectable, fd, event)