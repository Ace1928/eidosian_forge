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
def connection_closed(self, connection):
    """This is a Pyngus callback, invoked by Pyngus when the connection has
        cleanly closed.  This occurs after the driver closes the connection
        locally, and the peer has acknowledged the close.  At this point, the
        shutdown of the driver's connection is complete.
        """
    LOG.debug('AMQP connection closed.')
    self._handle_connection_loss('AMQP connection closed.')