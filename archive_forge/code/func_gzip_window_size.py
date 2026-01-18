from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def gzip_window_size(self):
    if self._values['gzip_window_size'] is None:
        return None
    return self._values['gzip_window_size'] / 1024