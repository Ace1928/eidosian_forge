from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def assigned_role(self):
    if self._values['assigned_role'] is None:
        return None
    rmap = dict(((v, k) for k, v in iteritems(self.role_map)))
    return rmap.get(self._values['assigned_role'], self._values['assigned_role'])