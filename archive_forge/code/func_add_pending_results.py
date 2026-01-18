import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
def add_pending_results(self, results, weak=False):
    self.result_consumer.drainer.start()
    return [self.add_pending_result(result, weak=weak, start_drainer=False) for result in results]