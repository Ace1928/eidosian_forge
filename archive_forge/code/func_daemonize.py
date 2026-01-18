import errno
import os
import pwd
import sys
import traceback
from twisted import copyright, logger
from twisted.application import app, service
from twisted.internet.interfaces import IReactorDaemonize
from twisted.python import log, logfile, usage
from twisted.python.runtime import platformType
from twisted.python.util import gidFromString, switchUID, uidFromString, untilConcludes
def daemonize(self, reactor):
    """
        Daemonizes the application on Unix. This is done by the usual double
        forking approach.

        @see: U{http://code.activestate.com/recipes/278731/}
        @see: W. Richard Stevens,
            "Advanced Programming in the Unix Environment",
            1992, Addison-Wesley, ISBN 0-201-56317-7

        @param reactor: The reactor in use.  If it provides
            L{IReactorDaemonize}, its daemonization-related callbacks will be
            invoked.

        @return: A writable pipe to be used to report errors.
        @rtype: C{int}
        """
    if IReactorDaemonize.providedBy(reactor):
        reactor.beforeDaemonize()
    r, w = os.pipe()
    if os.fork():
        code = self._waitForStart(r)
        os.close(r)
        os._exit(code)
    os.setsid()
    if os.fork():
        os._exit(0)
    null = os.open('/dev/null', os.O_RDWR)
    for i in range(3):
        try:
            os.dup2(null, i)
        except OSError as e:
            if e.errno != errno.EBADF:
                raise
    os.close(null)
    if IReactorDaemonize.providedBy(reactor):
        reactor.afterDaemonize()
    return w