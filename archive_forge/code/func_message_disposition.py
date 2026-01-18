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
def message_disposition(released=False):
    if receiver in self._receivers and (not receiver.closed):
        if released:
            receiver.message_released(handle)
        else:
            receiver.message_accepted(handle)
        if receiver.capacity <= self._capacity_low:
            receiver.add_capacity(self._capacity - receiver.capacity)
    else:
        LOG.debug("Can't find receiver for settlement")