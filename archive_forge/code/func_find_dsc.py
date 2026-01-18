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
def find_dsc(self):
    conn = http.client.HTTPConnection('localhost', DISCOVERY_CLIENT_PORT)
    try:
        conn.request('HEAD', '/metrics')
        resp = conn.getresponse()
        return 'found' if resp.status == http.client.OK else ''
    except Exception as e:
        LOG.debug('LIGHTOS: %s', e)
        out = ''
    return out