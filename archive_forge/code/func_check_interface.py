from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def check_interface(module, netcfg):
    config = str(netcfg)
    has_interface = re.search('(?:interface nve)(?P<value>.*)$', config, re.M)
    value = ''
    if has_interface:
        value = 'nve{0}'.format(has_interface.group('value'))
    return value