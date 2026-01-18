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
def test_graceful_stop_with_exceeded_graceful_shutdown_timeout(self):
    graceful_shutdown_timeout = 4
    self.config(graceful_shutdown_timeout=graceful_shutdown_timeout)
    proc, conn = self.run_server()
    time_before = time.time()
    os.kill(proc.pid, signal.SIGTERM)
    self.assertTrue(proc.is_alive())
    proc.join()
    self.assertFalse(proc.is_alive())
    time_after = time.time()
    self.assertTrue(time_after - time_before > graceful_shutdown_timeout)