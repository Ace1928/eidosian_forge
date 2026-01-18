import os
import signal
import threading
import time
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import exception
from os_brick import privileged
def on_execute(proc):
    if on_execute_call:
        on_execute_call(proc)
    if shared_data[0] and interval:
        exp = backoff_rate ** shared_data[0]
        wait_for = max(0, interval * exp)
        LOG.debug('Sleeping for %s seconds', wait_for)
        time.sleep(wait_for)
    shared_data[0] += 1
    if timeout:
        shared_data[2] = None
        shared_data[1] = threading.Timer(timeout, on_timeout, (proc,))
        shared_data[1].start()