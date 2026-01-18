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
def sasl_done(self, connection, pn_sasl, outcome):
    """This is a Pyngus callback invoked when the SASL handshake
        has completed.  The outcome of the handshake is passed in the outcome
        argument.
        """
    if outcome == proton.SASL.OK:
        return
    LOG.error('AUTHENTICATION FAILURE: Cannot connect to %(hostname)s:%(port)s as user %(username)s', {'hostname': self.hosts.current.hostname, 'port': self.hosts.current.port, 'username': self.hosts.current.username})