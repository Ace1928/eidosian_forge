import logging
import os
import string
import sys
import time
from taskflow import engines
from taskflow.engines.worker_based import worker
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import notifier
from taskflow.utils import threading_utils
def event_receiver(event_type, details):
    """This is the callback that (in this example) doesn't do much..."""
    print("Recieved event '%s'" % event_type)
    print('Details = %s' % details)