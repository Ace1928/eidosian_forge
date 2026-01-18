import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def execute_retry(self, retry, arguments):
    """Schedules retry execution."""
    fut = self._executor.submit(_execute_retry, retry, arguments)
    fut.atom = retry
    return fut