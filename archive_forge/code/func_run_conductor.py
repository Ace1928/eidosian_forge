import contextlib
import functools
import logging
import os
import sys
import time
import traceback
from kazoo import client
from taskflow.conductors import backends as conductor_backends
from taskflow import engines
from taskflow.jobs import backends as job_backends
from taskflow import logging as taskflow_logging
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends as persistence_backends
from taskflow.persistence import models
from taskflow import task
from oslo_utils import timeutils
from oslo_utils import uuidutils
def run_conductor(only_run_once=False):
    event_watches = {}

    def on_conductor_event(cond, event, details):
        print("Event '%s' has been received..." % event)
        print('Details = %s' % details)
        if event.endswith('_start'):
            w = timeutils.StopWatch()
            w.start()
            base_event = event[0:-len('_start')]
            event_watches[base_event] = w
        if event.endswith('_end'):
            base_event = event[0:-len('_end')]
            try:
                w = event_watches.pop(base_event)
                w.stop()
                print("It took %0.3f seconds for event '%s' to finish" % (w.elapsed(), base_event))
            except KeyError:
                pass
        if event == 'running_end' and only_run_once:
            cond.stop()
    print('Starting conductor with pid: %s' % ME)
    my_name = 'conductor-%s' % ME
    persist_backend = persistence_backends.fetch(PERSISTENCE_URI)
    with contextlib.closing(persist_backend):
        with contextlib.closing(persist_backend.get_connection()) as conn:
            conn.upgrade()
        job_backend = job_backends.fetch(my_name, JB_CONF, persistence=persist_backend)
        job_backend.connect()
        with contextlib.closing(job_backend):
            cond = conductor_backends.fetch('blocking', my_name, job_backend, persistence=persist_backend)
            on_conductor_event = functools.partial(on_conductor_event, cond)
            cond.notifier.register(cond.notifier.ANY, on_conductor_event)
            try:
                cond.run()
            finally:
                cond.stop()
                cond.wait()