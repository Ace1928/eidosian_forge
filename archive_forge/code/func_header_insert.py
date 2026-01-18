from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def header_insert(self):
    if self.want.header_insert is None:
        return None
    if self.want.header_insert in ['none', '']:
        if self.have.header_insert in [None, 'none']:
            return None
    if self.want.header_insert != self.have.header_insert:
        return self.want.header_insert