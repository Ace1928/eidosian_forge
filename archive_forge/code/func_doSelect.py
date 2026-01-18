import select
import sys
from errno import EBADF, EINTR
from time import sleep
from typing import Type
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
from twisted.python.runtime import platformType
def doSelect(self, timeout):
    """
        Run one iteration of the I/O monitor loop.

        This will run all selectables who had input or output readiness
        waiting for them.
        """
    try:
        r, w, ignored = _select(self._reads, self._writes, [], timeout)
    except ValueError:
        self._preenDescriptors()
        return
    except TypeError:
        log.err()
        self._preenDescriptors()
        return
    except OSError as se:
        if se.args[0] in (0, 2):
            if not self._reads and (not self._writes):
                return
            else:
                raise
        elif se.args[0] == EINTR:
            return
        elif se.args[0] == EBADF:
            self._preenDescriptors()
            return
        else:
            raise
    _drdw = self._doReadOrWrite
    _logrun = log.callWithLogger
    for selectables, method, fdset in ((r, 'doRead', self._reads), (w, 'doWrite', self._writes)):
        for selectable in selectables:
            if selectable not in fdset:
                continue
            _logrun(selectable, _drdw, selectable, method)