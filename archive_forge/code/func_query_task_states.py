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
def query_task_states(self, ids, timeout=0.5):
    states = defaultdict(set)
    for hostname, reply in self.query_tasks(ids, timeout=timeout):
        for task_id, (state, _) in reply.items():
            states[state].add(task_id)
    return states