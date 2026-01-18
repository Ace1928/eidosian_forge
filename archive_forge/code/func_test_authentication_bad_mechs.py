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
def test_authentication_bad_mechs(self):
    """Verify that the connection fails if the client's SASL mechanisms do
        not match the broker's.
        """
    self.config(sasl_mechanisms='EXTERNAL ANONYMOUS', group='oslo_messaging_amqp')
    addr = 'amqp://joe@myrealm:secret@%s:%d' % (self._broker.host, self._broker.port)
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, self._authentication_test, addr, retry=0)