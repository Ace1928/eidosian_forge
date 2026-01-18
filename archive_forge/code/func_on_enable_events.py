import collections
import logging
import shelve
import threading
import time
from collections import Counter
from functools import partial
from celery.events import EventReceiver
from celery.events.state import State
from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Gauge, Histogram
from tornado.ioloop import PeriodicCallback
from tornado.options import options
def on_enable_events(self):
    self.io_loop.run_in_executor(None, self.capp.control.enable_events)