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
def test_killed_worker_recover(self):
    start_workers = self._spawn()
    LOG.info('pid of first child is %s' % start_workers[0])
    os.kill(start_workers[0], signal.SIGTERM)
    cond = lambda: start_workers != self._get_workers()
    timeout = 5
    self._wait(cond, timeout)
    end_workers = self._get_workers()
    LOG.info('workers: %r' % end_workers)
    self.assertNotEqual(start_workers, end_workers)