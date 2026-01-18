import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def test_heartbeat_failover(self):
    """Simulate broker heartbeat timeout."""

    def _meth(broker):
        broker.pause()
    self.config(idle_timeout=2, group='oslo_messaging_amqp')
    self._failover(_meth)
    self._brokers[self._primary].stop()