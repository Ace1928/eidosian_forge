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
def test_graceful_shuts_down_on_sigterm_when_client_connected(self):
    self.config(graceful_shutdown_timeout=7)
    proc, conn = self.run_server()
    os.kill(proc.pid, signal.SIGTERM)
    time.sleep(1)
    self.assertTrue(proc.is_alive())
    conn.close()
    proc.join()