from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.urls import urlparse
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def retain_package_file(self):
    return flatten_boolean(self._values['retain_package_file'])