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
def test_client_auth_ok(self):
    self._ssl_config['authenticate_client'] = True
    self._broker = FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=self._ssl_config['s_name'], ssl_config=self._ssl_config)
    url = oslo_messaging.TransportURL.parse(self.conf, 'amqp://%s:%d' % (self._broker.host, self._broker.port))
    self._broker.start()
    self.config(ssl_ca_file=self._ssl_config['ca_cert'], ssl_cert_file=self._ssl_config['c_cert'], ssl_key_file=self._ssl_config['c_key'], ssl_key_password=self._ssl_config['pw'], group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
    driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'a': 'b'}, wait_for_reply=True, timeout=30)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    driver.cleanup()