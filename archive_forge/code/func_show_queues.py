import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size
import click
from redis import Redis
from redis.sentinel import Sentinel
from rq.defaults import (
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus
def show_queues(queues, raw, by_queue, queue_class, worker_class):
    num_jobs = 0
    termwidth = get_terminal_size().columns
    chartwidth = min(20, termwidth - 20)
    max_count = 0
    counts = dict()
    for q in queues:
        count = q.count
        counts[q] = count
        max_count = max(max_count, count)
    scale = get_scale(max_count)
    ratio = chartwidth * 1.0 / scale
    for q in queues:
        count = counts[q]
        if not raw:
            chart = green('|' + '█' * int(ratio * count))
            line = '%-12s %s %d, %d executing, %d finished, %d failed' % (q.name, chart, count, q.started_job_registry.count, q.finished_job_registry.count, q.failed_job_registry.count)
        else:
            line = 'queue %s %d, %d executing, %d finished, %d failed' % (q.name, count, q.started_job_registry.count, q.finished_job_registry.count, q.failed_job_registry.count)
        click.echo(line)
        num_jobs += count
    if not raw:
        click.echo('%d queues, %d jobs total' % (len(queues), num_jobs))