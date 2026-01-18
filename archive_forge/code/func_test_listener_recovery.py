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
def test_listener_recovery(self):
    self._addrs = {'unicast.test-topic': 2, 'broadcast.test-topic.all': 2, 'exclusive.test-topic.server': 2}
    self._recovered = eventletutils.Event()
    self._count = 0

    def _on_active(link):
        t = link.target_address
        if t in self._addrs:
            if self._addrs[t] > 0:
                link.close()
                self._addrs[t] -= 1
            else:
                self._count += 1
                if self._count == len(self._addrs):
                    self._recovered.set()
    self._broker.on_sender_active = _on_active
    self._broker.start()
    self.config(link_retry_delay=1, group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic', server='server')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 3)
    self.assertTrue(self._recovered.wait(timeout=30))
    rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'e1'}, wait_for_reply=True)
    self.assertIsNotNone(rc)
    self.assertEqual(rc.get('correlation-id'), 'e1')
    target.server = None
    rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'e2'}, wait_for_reply=True)
    self.assertIsNotNone(rc)
    self.assertEqual(rc.get('correlation-id'), 'e2')
    target.fanout = True
    driver.send(target, {'context': 'whatever'}, {'msg': 'value'}, wait_for_reply=False)
    listener.join(timeout=30)
    self.assertTrue(self._broker.fanout_count == 1)
    self.assertFalse(listener.is_alive())
    self.assertEqual(listener.messages.get().message.get('method'), 'echo')
    driver.cleanup()