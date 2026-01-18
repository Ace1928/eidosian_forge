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
def connection_remote_closed(self, connection, reason):
    """This is a Pyngus callback, invoked by Pyngus when the peer has
        requested that the connection be closed.
        """
    if reason:
        LOG.info('Connection closed by peer: %s', reason)
    self._detach_senders()
    self._detach_servers()
    self.reply_link.detach()
    self._socket_connection.pyngus_conn.close()