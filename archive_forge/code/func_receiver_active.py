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
def receiver_active(self, receiver_link):
    """This is a Pyngus callback, invoked by Pyngus when the receiver_link
        has transitioned to the open state and is able to receive incoming
        messages.
        """
    LOG.debug('Replies link active src=%s', self._receiver.source_address)
    receiver_link.add_capacity(self._capacity)
    self._on_ready()