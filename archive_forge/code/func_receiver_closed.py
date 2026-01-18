import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
def receiver_closed(self, receiver_link):
    LOG.debug('Server subscription to %s closed', receiver_link.source_address)
    if self._connection and (not self._reopen_scheduled):
        LOG.debug('Server subscription reopen scheduled')
        self._reopen_scheduled = True
        self._scheduler.defer(self._reopen_links, self._delay)