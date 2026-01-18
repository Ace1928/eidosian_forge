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
def receiver_remote_closed(self, receiver, pn_condition):
    """This is a Pyngus callback, invoked by Pyngus when the peer of this
        receiver link has initiated closing the connection.
        """
    LOG.debug('Server subscription to %s remote detach', receiver.source_address)
    if pn_condition:
        vals = {'addr': receiver.source_address or receiver.target_address, 'err_msg': pn_condition}
        LOG.error('Server subscription %(addr)s closed by peer: %(err_msg)s', vals)
    receiver.close()