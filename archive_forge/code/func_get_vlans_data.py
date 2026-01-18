from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.vlans.vlans import (
def get_vlans_data(self, connection, configuration):
    """Checks device is L2/L3 and returns
        facts gracefully. Does not fail module.
        """
    if configuration:
        cmd = 'show running-config | sec ^vlan configuration .+'
    else:
        cmd = 'show vlan'
    check_os_type = connection.get_device_info()
    if check_os_type.get('network_os_type') == 'L3':
        return ''
    return connection.get(cmd)