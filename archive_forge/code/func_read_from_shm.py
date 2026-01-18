import logging
import os
import queue
import threading
import time
import uuid
import cachetools
from oslo_concurrency import lockutils
from oslo_utils import eventletutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging import MessageDeliveryFailure
@lockutils.synchronized(lock_name, external=True)
def read_from_shm():
    try:
        with open(self.file_name, 'r') as f:
            pg, c = f.readline().split(':')
            pg = int(pg)
            c = int(c)
    except (FileNotFoundError, ValueError):
        pg = self.pg
        c = 0
    if pg == self.pg:
        c += 1
    else:
        c = 1
    with open(self.file_name, 'w') as f:
        f.write(str(self.pg) + ':' + str(c))
    return c