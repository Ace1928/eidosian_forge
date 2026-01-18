from __future__ import absolute_import, division, print_function
import os
import re
import socket
import ssl
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def max_file_size(self):
    if self._values['max_file_size'] in [None]:
        return None
    return '-s {0}'.format(self._values['max_file_size'])