import socket
import sys
from collections import defaultdict
from functools import partial
from itertools import count
from typing import Any, Callable, Dict, Sequence, TextIO, Tuple  # noqa
from kombu.exceptions import ContentDisallowed
from kombu.utils.functional import retry_over_time
from celery import states
from celery.exceptions import TimeoutError
from celery.result import AsyncResult, ResultSet  # noqa
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds as _humanize_seconds
def assert_result_tasks_in_progress_or_completed(self, async_results, interval=0.5, desc='waiting for tasks to be started or completed', **policy):
    return self.assert_task_state_from_result(self.is_result_task_in_progress, async_results, interval=interval, desc=desc, **policy)