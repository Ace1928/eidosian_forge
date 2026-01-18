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
def monitor_db(self, lightos_db):
    for connection_info in lightos_db.values():
        self.dsc_connect_volume(connection_info)