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
@mock.patch('oslo_service.service.ProcessLauncher._start_child')
@mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
@mock.patch('eventlet.greenio.GreenPipe')
@mock.patch('os.pipe')
def test_check_service_base(self, pipe_mock, green_pipe_mock, handle_signal_mock, start_child_mock):
    pipe_mock.return_value = [None, None]
    launcher = service.ProcessLauncher(self.conf)
    serv = _Service()
    launcher.launch_service(serv, workers=0)