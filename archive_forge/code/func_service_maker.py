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
def service_maker():
    self.conf.register_mutate_hook(lambda c, f: mutate.set())
    return ServiceWithTimer(ready)