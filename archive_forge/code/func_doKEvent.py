import errno
import select
from zope.interface import Attribute, Interface, declarations, implementer
from twisted.internet import main, posixbase
from twisted.internet.interfaces import IReactorDaemonize, IReactorFDSet
from twisted.python import failure, log
def doKEvent(self, timeout):
    """
        Poll the kqueue for new events.
        """
    if timeout is None:
        timeout = 1
    try:
        events = self._kq.control([], len(self._selectables), timeout)
    except OSError as e:
        if e.errno == errno.EINTR:
            return
        else:
            raise
    _drdw = self._doWriteOrRead
    for event in events:
        fd = event.ident
        try:
            selectable = self._selectables[fd]
        except KeyError:
            continue
        else:
            log.callWithLogger(selectable, _drdw, selectable, fd, event)