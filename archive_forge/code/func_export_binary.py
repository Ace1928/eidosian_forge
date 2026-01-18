from __future__ import absolute_import, division, print_function
import os
import time
import tempfile
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def export_binary(self):
    self.export_binary_on_device()
    self.execute()
    return True