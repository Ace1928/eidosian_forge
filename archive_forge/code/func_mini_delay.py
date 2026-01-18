import time
import eventlet
import testscenarios
import futurist
from futurist.tests import base
from futurist import waiters
def mini_delay(use_eventlet_sleep=False):
    if use_eventlet_sleep:
        eventlet.sleep(0.1)
    else:
        time.sleep(0.1)
    return 1