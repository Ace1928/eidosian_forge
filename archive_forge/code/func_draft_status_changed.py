from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def draft_status_changed(self):
    if self.draft_exists() and self.want.state == 'draft':
        drafted = False
    elif not self.draft_exists() and self.want.state == 'present':
        drafted = False
    else:
        drafted = True
    return drafted