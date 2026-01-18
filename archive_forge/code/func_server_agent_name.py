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
def server_agent_name(self):
    if self.want.server_agent_name is None:
        return None
    if self.want.server_agent_name in ['none', '']:
        if self.have.server_agent_name in [None, 'none']:
            return None
    if self.want.server_agent_name != self.have.server_agent_name:
        return self.want.server_agent_name