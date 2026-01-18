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
def test_authentication_failure(self):
    """Verify that a bad password given in TransportHost is
        rejected by the broker.
        """
    addr = 'amqp://joe@myrealm:badpass@%s:%d' % (self._broker.host, self._broker.port)
    try:
        self._authentication_test(addr, retry=2)
    except oslo_messaging.MessageDeliveryFailure as e:
        self.assertTrue('amqp:unauthorized-access' in str(e))
    else:
        self.assertIsNone('Expected authentication failure')