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
def sender_requested(self, connection, link_handle, name, requested_source, properties):
    """Create a new message source."""
    addr = requested_source or 'source-' + uuid.uuid4().hex
    link = FakeBroker.SenderLink(self.server, self, link_handle, addr)
    self.sender_links.add(link)