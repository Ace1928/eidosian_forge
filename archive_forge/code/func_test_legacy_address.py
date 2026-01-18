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
def test_legacy_address(self):
    self.config(addressing_mode='legacy', group='oslo_messaging_amqp')
    _opts = self.conf.oslo_messaging_amqp
    notifications = [(oslo_messaging.Target(topic='test-topic'), 'info'), (oslo_messaging.Target(topic='test-topic'), 'error'), (oslo_messaging.Target(topic='test-topic'), 'debug')]
    msgs = self._address_test(oslo_messaging.Target(exchange='ex', topic='test-topic'), notifications)
    addrs = [m.address for m in msgs]
    server_addrs = [a for a in addrs if a.startswith(_opts.server_request_prefix)]
    broadcast_addrs = [a for a in addrs if a.startswith(_opts.broadcast_prefix)]
    group_addrs = [a for a in addrs if a.startswith(_opts.group_request_prefix)]
    self.assertEqual(len(server_addrs), 2)
    self.assertEqual(len(broadcast_addrs), 1)
    self.assertEqual(len(group_addrs), 2 + len(notifications))