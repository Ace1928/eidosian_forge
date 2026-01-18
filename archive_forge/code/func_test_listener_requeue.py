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
def test_listener_requeue(self):
    """Emulate Server requeue on listener incoming messages"""
    self.config(pre_settled=[], group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    driver.require_features(requeue=True)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1, msg_ack=False)
    rc = driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
    self.assertIsNone(rc)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    predicate = lambda: self._broker.sender_link_requeue_count == 1
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    driver.cleanup()