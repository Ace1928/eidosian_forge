import os
import platform
import shelve
import sys
import weakref
import zlib
from collections import Counter
from kombu.serialization import pickle, pickle_protocol
from kombu.utils.objects import cached_property
from celery import __version__
from celery.exceptions import WorkerShutdown, WorkerTerminate
from celery.utils.collections import LimitedSet
def task_accepted(request, _all_total_count=None, add_request=requests.__setitem__, add_active_request=active_requests.add, add_to_total_count=total_count.update):
    """Update global state when a task has been accepted."""
    if not _all_total_count:
        _all_total_count = all_total_count
    add_request(request.id, request)
    add_active_request(request)
    add_to_total_count({request.name: 1})
    all_total_count[0] += 1