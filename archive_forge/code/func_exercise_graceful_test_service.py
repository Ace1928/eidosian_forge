import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
def exercise_graceful_test_service(sleep_amount, time_to_wait, graceful):
    svc = GracefulShutdownTestService()
    svc.start(sleep_amount)
    svc.stop(graceful)

    def wait_for_task(svc):
        svc.finished_task.wait()
    return eventlet.timeout.with_timeout(time_to_wait, wait_for_task, svc=svc, timeout_value='Timeout!')