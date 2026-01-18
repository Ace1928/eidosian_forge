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
def ensure_not_for_a_while(self, fun, catch, desc='thing', max_retries=20, interval_start=0.1, interval_step=0.02, interval_max=1.0, emit_warning=False, **options):
    """Make sure something does not happen (at least for a while)."""
    try:
        return self.wait_for(fun, catch, desc=desc, max_retries=max_retries, interval_start=interval_start, interval_step=interval_step, interval_max=interval_max, emit_warning=emit_warning)
    except catch:
        pass
    else:
        raise AssertionError(f'Should not have happened: {desc}')