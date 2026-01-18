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
def lightos_monitor(self, lightos_db, message_queue):
    """Bookkeeping lightos connections.

        This is useful when the connector is comming up to a running node with
        connected volumes already exists.
        This is used in the Nova driver to restore connections after reboot
        """
    first_time = True
    while True:
        self.monitor_db(lightos_db)
        if first_time:
            time.sleep(5)
            first_time = False
        else:
            time.sleep(1)
        self.monitor_message_queue(message_queue, lightos_db)