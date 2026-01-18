from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def dag_ipv6_prefix_len(self):
    if self._values['dag'] is None:
        return None
    if self._values['dag']['dag_ipv6_prefix_len'] is None:
        return None
    if 0 <= self._values['dag']['dag_ipv6_prefix_len'] <= 128:
        return self._values['dag']['dag_ipv6_prefix_len']
    raise F5ModuleError("Valid 'dag_ipv6_prefix_len' must be in range 0 - 128.")