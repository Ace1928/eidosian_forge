import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def run_to_completion(self, wait_time=1, progress_callback=None):
    """Run the task to completion.

        The task will sleep for `wait_time` seconds between steps. To avoid
        sleeping, pass `None` for `wait_time`.
        """
    assert self._runner is not None, 'Task not started'
    for step in self.as_task(progress_callback=progress_callback):
        self._sleep(wait_time)