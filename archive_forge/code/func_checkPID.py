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
def checkPID(pidfile):
    if not pidfile:
        return
    if os.path.exists(pidfile):
        try:
            with open(pidfile) as f:
                pid = int(f.read())
        except ValueError:
            sys.exit(f'Pidfile {pidfile} contains non-numeric value')
        try:
            os.kill(pid, 0)
        except OSError as why:
            if why.errno == errno.ESRCH:
                log.msg(f'Removing stale pidfile {pidfile}', isError=True)
                os.remove(pidfile)
            else:
                sys.exit("Can't check status of PID {} from pidfile {}: {}".format(pid, pidfile, why))
        else:
            sys.exit('Another twistd server is running, PID {}\n\nThis could either be a previously started instance of your application or a\ndifferent application entirely. To start a new one, either run it in some other\ndirectory, or use the --pidfile and --logfile parameters to avoid clashes.\n'.format(pid))