import collections
import contextlib
import threading
from fasteners import _utils
import six
@contextlib.contextmanager
def read_lock(self):
    """Context manager that grants a read lock.

        Will wait until no active or pending writers.

        Raises a ``RuntimeError`` if a pending writer tries to acquire
        a read lock.
        """
    me = self._current_thread()
    if me in self._pending_writers:
        raise RuntimeError('Writer %s can not acquire a read lock while waiting for the write lock' % me)
    with self._cond:
        while True:
            if self._writer is None or self._writer == me:
                try:
                    self._readers[me] = self._readers[me] + 1
                except KeyError:
                    self._readers[me] = 1
                break
            self._cond.wait()
    try:
        yield self
    finally:
        with self._cond:
            try:
                me_instances = self._readers[me]
                if me_instances > 1:
                    self._readers[me] = me_instances - 1
                else:
                    self._readers.pop(me)
            except KeyError:
                pass
            self._cond.notify_all()