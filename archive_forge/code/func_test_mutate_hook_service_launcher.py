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
def test_mutate_hook_service_launcher(self):
    """Test mutate_config_files is called by ServiceLauncher on SIGHUP.

        Not using _spawn_service because ServiceLauncher doesn't fork and it's
        simplest to stay all in one process.
        """
    mutate = multiprocessing.Event()
    self.conf.register_mutate_hook(lambda c, f: mutate.set())
    launcher = service.launch(self.conf, ServiceWithTimer(), restart_method='mutate')
    self.assertFalse(mutate.is_set(), 'Hook was called too early')
    launcher.restart()
    self.assertTrue(mutate.is_set(), "Hook wasn't called")