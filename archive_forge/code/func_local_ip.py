from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def local_ip(self):
    if self._values['local_ip'] in [None, 'none']:
        return None
    if is_valid_ip(self._values['local_ip']):
        return self._values['local_ip']
    else:
        raise F5ModuleError("The provided 'local_ip' is not a valid IP address")