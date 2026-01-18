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
def test_listener_failover(self):
    """Verify that Listeners sharing the same topic are re-established
        after failover.
        """
    self._brokers[0].start()
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='my-topic')
    bcast = oslo_messaging.Target(topic='my-topic', fanout=True)
    listener1 = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 2)
    listener2 = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 2)
    predicate = lambda: self._brokers[0].sender_link_count == 7
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    driver.send(bcast, {'context': 'whatever'}, {'method': 'ignore', 'id': 'echo-1'})
    predicate = lambda: self._brokers[0].fanout_sent_count == 2
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    self._brokers[1].start()
    self._brokers[0].stop(clean=True)
    predicate = lambda: self._brokers[1].sender_link_count == 7
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    driver.send(bcast, {'context': 'whatever'}, {'method': 'ignore', 'id': 'echo-2'})
    predicate = lambda: self._brokers[1].fanout_sent_count == 2
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    listener1.join(timeout=30)
    listener2.join(timeout=30)
    self.assertFalse(listener1.is_alive() or listener2.is_alive())
    driver.cleanup()
    self._brokers[1].stop()