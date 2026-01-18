import sys
import time
from threading import Thread
from weakref import WeakKeyDictionary
from zope.interface import implementer
from win32file import FD_ACCEPT, FD_CLOSE, FD_CONNECT, FD_READ, WSAEventSelect
import win32gui  # type: ignore[import-untyped]
from win32event import (
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet, IReactorWin32Events
from twisted.internet.threads import blockingCallFromThread
from twisted.python import failure, log, threadable
def doWaitForMultipleEvents(self, timeout):
    log.msg(channel='system', event='iteration', reactor=self)
    if timeout is None:
        timeout = 100
    ranUserCode = False
    for reader in list(self._closedAndReading.keys()):
        ranUserCode = True
        self._runAction('doRead', reader)
    for fd in list(self._writes.keys()):
        ranUserCode = True
        log.callWithLogger(fd, self._runWrite, fd)
    if ranUserCode:
        timeout = 0
    if not (self._events or self._writes):
        time.sleep(timeout)
        return
    handles = list(self._events.keys()) or [self.dummyEvent]
    timeout = int(timeout * 1000)
    val = MsgWaitForMultipleObjects(handles, 0, timeout, QS_ALLINPUT)
    if val == WAIT_TIMEOUT:
        return
    elif val == WAIT_OBJECT_0 + len(handles):
        exit = win32gui.PumpWaitingMessages()
        if exit:
            self.callLater(0, self.stop)
            return
    elif val >= WAIT_OBJECT_0 and val < WAIT_OBJECT_0 + len(handles):
        event = handles[val - WAIT_OBJECT_0]
        fd, action = self._events[event]
        if fd in self._reads:
            fileno = fd.fileno()
            if fileno == -1:
                self._disconnectSelectable(fd, posixbase._NO_FILEDESC, False)
                return
            events = WSAEnumNetworkEvents(fileno, event)
            if FD_CLOSE in events:
                self._closedAndReading[fd] = True
        log.callWithLogger(fd, self._runAction, action, fd)