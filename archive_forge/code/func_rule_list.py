from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def rule_list(self):
    if self._values['rule_list'] is None:
        return None
    if self._values['parent_policy'] is not None:
        return fq_name(self.partition, self._values['rule_list'])
    return None