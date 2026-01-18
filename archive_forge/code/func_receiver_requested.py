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
def receiver_requested(self, connection, link_handle, name, requested_target, properties):
    """Create a new message consumer."""
    addr = requested_target or 'target-' + uuid.uuid4().hex
    FakeBroker.ReceiverLink(self.server, self, link_handle, addr)