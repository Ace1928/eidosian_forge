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
def config_sync_ip(self):
    if self._values['config_sync_ip'] is None:
        return None
    elif self._values['config_sync_ip'] in ['none', '']:
        return 'none'
    result = self._get_validated_ip_address('config_sync_ip')
    return result