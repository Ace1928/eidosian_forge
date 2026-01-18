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
def test_call_late_reply(self):
    """What happens if reply arrives after timeout?"""
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _SlowResponder(driver.listen(target, None, None)._poll_style_listener, delay=3)
    self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': 'whatever'}, {'method': 'echo', 'id': '???'}, wait_for_reply=True, timeout=1.0)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    predicate = lambda: self._broker.sender_link_ack_count == 1
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    driver.cleanup()