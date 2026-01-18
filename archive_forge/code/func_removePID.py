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
def removePID(self, pidfile):
    """
        Remove the specified PID file, if possible.  Errors are logged, not
        raised.

        @type pidfile: C{str}
        @param pidfile: The path to the PID tracking file.
        """
    if not pidfile:
        return
    try:
        os.unlink(pidfile)
    except OSError as e:
        if e.errno == errno.EACCES or e.errno == errno.EPERM:
            log.msg('Warning: No permission to delete pid file')
        else:
            log.err(e, 'Failed to unlink PID file:')
    except BaseException:
        log.err(None, 'Failed to unlink PID file:')