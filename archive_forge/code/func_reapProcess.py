from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def reapProcess(self):
    """
        Try to reap a process (without blocking) via waitpid.

        This is called when sigchild is caught or a Process object loses its
        "connection" (stdout is closed) This ought to result in reaping all
        zombie processes, since it will be called twice as often as it needs
        to be.

        (Unfortunately, this is a slightly experimental approach, since
        UNIX has no way to be really sure that your process is going to
        go away w/o blocking.  I don't want to block.)
        """
    try:
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except OSError as e:
            if e.errno == errno.ECHILD:
                pid = None
            else:
                raise
    except BaseException:
        log.msg(f'Failed to reap {self.pid}:')
        log.err()
        pid = None
    if pid:
        unregisterReapProcessHandler(pid, self)
        self.processEnded(status)