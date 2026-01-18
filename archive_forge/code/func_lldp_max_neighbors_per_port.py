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
def lldp_max_neighbors_per_port(self):
    if self._values['lldp'] is None:
        return None
    if self._values['lldp']['max_neighbors_per_port'] is None:
        return None
    if 0 <= self._values['lldp']['max_neighbors_per_port'] <= 65535:
        return self._values['lldp']['max_neighbors_per_port']
    raise F5ModuleError("Valid 'max_neighbors_per_port' must be in range 0 - 65535.")