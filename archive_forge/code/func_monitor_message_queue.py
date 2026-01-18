import glob
import http.client
import os
import re
import tempfile
import time
import traceback
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import lightos as priv_lightos
from os_brick import utils
def monitor_message_queue(self, message_queue, lightos_db):
    while not message_queue.empty():
        msg = message_queue.get()
        op, connection = msg
        LOG.debug('LIGHTOS: queue got op: %s, connection: %s', op, connection)
        if op == 'delete':
            LOG.info('LIGHTOS: Removing volume: %s from db', connection['uuid'])
            if connection['uuid'] in lightos_db:
                del lightos_db[connection['uuid']]
            else:
                LOG.warning('LIGHTOS: No volume: %s found in db', connection['uuid'])
        elif op == 'add':
            LOG.info('LIGHTOS: Adding volume: %s to db', connection['uuid'])
            lightos_db[connection['uuid']] = connection