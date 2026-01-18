from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, execute_nc_action, ce_argument_spec
def config_vlan(self, vlan_id, name='', description=''):
    """Create vlan."""
    if name is None:
        name = ''
    if description is None:
        description = ''
    conf_str = CE_NC_CREATE_VLAN % (vlan_id, name, description)
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'CREATE_VLAN')
    self.changed = True