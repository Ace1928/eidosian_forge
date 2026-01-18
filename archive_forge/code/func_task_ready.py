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
def task_ready(request):
    """Called when a task is completed."""
    global all_count
    global bench_start
    global bench_last
    all_count += 1
    if not all_count % bench_every:
        now = monotonic()
        diff = now - bench_start
        print('- Time spent processing {} tasks (since first task received): ~{:.4f}s\n'.format(bench_every, diff))
        sys.stdout.flush()
        bench_start = bench_last = now
        bench_sample.append(diff)
        sample_mem()
    return __ready(request)