from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def download_from_device(self, dest):
    url = 'https://{0}:{1}/mgmt/shared/file-transfer/ucs-downloads/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.src)
    try:
        download_file(self.client, url, dest)
    except F5ModuleError:
        raise F5ModuleError('Failed to download the file.')
    if os.path.exists(self.want.dest):
        return True
    return False