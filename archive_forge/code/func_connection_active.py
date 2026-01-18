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
def connection_active(self, connection):
    """This is a Pyngus callback, invoked by Pyngus when the connection to
        the peer is up.  At this point, the driver will activate all subscriber
        links (server) and the reply link.
        """
    LOG.debug('Connection active (%(hostname)s:%(port)s), subscribing...', {'hostname': self.hosts.current.hostname, 'port': self.hosts.current.port})
    props = connection.remote_properties or {}
    self.addresser = self.addresser_factory(props, self.hosts.virtual_host if self.pseudo_vhost else None)
    for servers in self._servers.values():
        for server in servers.values():
            server.attach(self._socket_connection.pyngus_conn, self.addresser)
    self.reply_link = Replies(self._socket_connection.pyngus_conn, self._reply_link_ready, self._reply_link_down, self._reply_credit)
    self._delay = self.conn_retry_interval
    self._link_maint_timer = self.processor.defer(self._purge_sender_links, self._link_maint_timeout)