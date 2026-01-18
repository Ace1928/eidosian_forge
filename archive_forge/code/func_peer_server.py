from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def peer_server(self):
    if self._values['peer_server'] is None:
        return None
    if is_valid_ip(self._values['peer_server']):
        return self._values['peer_server']
    raise F5ModuleError("The provided 'peer_server' parameter is not an IP address.")