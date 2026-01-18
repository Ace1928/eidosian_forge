from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def link_discovery(self):
    self._discovery_constraints()
    if self.want.link_discovery != self.have.link_discovery:
        return self.want.link_discovery