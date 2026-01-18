import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def as_task(self, timeout=None, progress_callback=None):
    """Return a task that drives the TaskRunner."""
    resuming = self.started()
    if not resuming:
        self.start(timeout=timeout)
    elif timeout is not None:
        new_timeout = Timeout(self, timeout)
        if new_timeout.earlier_than(self._timeout):
            self._timeout = new_timeout
    done = self.step() if resuming else self.done()
    while not done:
        try:
            yield
            if progress_callback is not None:
                progress_callback()
        except GeneratorExit:
            self.cancel()
            raise
        except:
            self._done = True
            try:
                self._runner.throw(*sys.exc_info())
            except StopIteration:
                return
            else:
                self._done = False
        else:
            done = self.step()