from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def create_blank(self):
    self.create_on_device()
    if self.exists():
        return True
    else:
        raise F5ModuleError('Failed to create ASM policy: {0}'.format(self.want.name))