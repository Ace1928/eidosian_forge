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
def test_messaging_patterns(self):
    """Verify the direct, shared, and fanout message patterns work."""
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target1 = oslo_messaging.Target(topic='test-topic', server='server1')
    listener1 = _ListenerThread(driver.listen(target1, None, None)._poll_style_listener, 4)
    target2 = oslo_messaging.Target(topic='test-topic', server='server2')
    listener2 = _ListenerThread(driver.listen(target2, None, None)._poll_style_listener, 3)
    shared_target = oslo_messaging.Target(topic='test-topic')
    fanout_target = oslo_messaging.Target(topic='test-topic', fanout=True)
    driver.send(shared_target, {'context': 'whatever'}, {'method': 'echo', 'id': 'either-1'}, wait_for_reply=True)
    self.assertEqual(1, self._broker.topic_count)
    self.assertEqual(1, self._broker.direct_count)
    driver.send(shared_target, {'context': 'whatever'}, {'method': 'echo', 'id': 'either-2'}, wait_for_reply=True)
    self.assertEqual(2, self._broker.topic_count)
    self.assertEqual(2, self._broker.direct_count)
    driver.send(target1, {'context': 'whatever'}, {'method': 'echo', 'id': 'server1-1'}, wait_for_reply=True)
    driver.send(target1, {'context': 'whatever'}, {'method': 'echo', 'id': 'server1-2'}, wait_for_reply=True)
    self.assertEqual(6, self._broker.direct_count)
    driver.send(target2, {'context': 'whatever'}, {'method': 'echo', 'id': 'server2'}, wait_for_reply=True)
    self.assertEqual(8, self._broker.direct_count)
    driver.send(fanout_target, {'context': 'whatever'}, {'method': 'echo', 'id': 'fanout'})
    listener1.join(timeout=30)
    self.assertFalse(listener1.is_alive())
    listener2.join(timeout=30)
    self.assertFalse(listener2.is_alive())
    self.assertEqual(1, self._broker.fanout_count)
    listener1_ids = [x.message.get('id') for x in listener1.get_messages()]
    listener2_ids = [x.message.get('id') for x in listener2.get_messages()]
    self.assertTrue('fanout' in listener1_ids and 'fanout' in listener2_ids)
    self.assertTrue('server1-1' in listener1_ids and 'server1-1' not in listener2_ids)
    self.assertTrue('server1-2' in listener1_ids and 'server1-2' not in listener2_ids)
    self.assertTrue('server2' in listener2_ids and 'server2' not in listener1_ids)
    if 'either-1' in listener1_ids:
        self.assertTrue('either-2' in listener2_ids and 'either-2' not in listener1_ids and ('either-1' not in listener2_ids))
    else:
        self.assertTrue('either-2' in listener1_ids and 'either-2' not in listener2_ids and ('either-1' in listener2_ids))
    predicate = lambda: self._broker.sender_link_ack_count == 12
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    driver.cleanup()