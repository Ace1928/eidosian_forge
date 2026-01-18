from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config, run_commands
def get_switchport(name, module):
    config = run_commands(module, ['show interface {0} switchport'.format(name)])[0]
    mode = re.search('Switchport mode\\s+:\\s(?:.* )?(\\w+)$', config, re.M)
    if mode:
        mode = mode.group(1)
    access = re.search('Default Vlan\\s+:\\s(\\d+)', config)
    if access:
        access = access.group(1)
    native = re.search('Native Vlan\\s+:\\s(\\d+)', config)
    if native:
        native = native.group(1)
    trunk = re.search('Active Vlans\\s+:\\s(.+)$', config, re.M)
    if trunk:
        trunk = trunk.group(1)
    if trunk == 'ALL':
        trunk = '1-4094'
    switchport_config = {'interface': name, 'mode': mode, 'access_vlan': access, 'native_vlan': native, 'trunk_vlans': trunk}
    return switchport_config