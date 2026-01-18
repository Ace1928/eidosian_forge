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
def forward_message(self, message, handle, rlink):
    self.message_log.append(message)
    dest = message.address
    if dest not in self._sources:
        self.dropped_count += 1
        if '!no-ack!' not in dest:
            rlink.message_released(handle)
        return
    LOG.debug('Forwarding [%s]', dest)
    if self._addresser._is_multicast(dest):
        self.fanout_count += 1
        for link in self._sources[dest]:
            self.fanout_sent_count += 1
            LOG.debug('Broadcast to %s', dest)
            link.send_message(message)
    elif self._addresser._is_anycast(dest):
        self.topic_count += 1
        link = self._sources[dest].pop(0)
        link.send_message(message)
        LOG.debug('Send to %s', dest)
        self._sources[dest].append(link)
    else:
        self.direct_count += 1
        LOG.debug('Unicast to %s', dest)
        self._sources[dest][0].send_message(message)
    rlink.message_accepted(handle)