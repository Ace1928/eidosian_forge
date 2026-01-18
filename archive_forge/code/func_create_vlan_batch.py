from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, execute_nc_action, ce_argument_spec
def create_vlan_batch(self, vlan_list):
    """Create vlan batch."""
    if not vlan_list:
        return
    vlan_bitmap = self.vlan_list_to_bitmap(vlan_list)
    xmlstr = CE_NC_CREATE_VLAN_BATCH % (vlan_bitmap, vlan_bitmap)
    recv_xml = execute_nc_action(self.module, xmlstr)
    self.check_response(recv_xml, 'CREATE_VLAN_BATCH')
    self.updates_cmd.append('vlan batch %s' % self.vlan_range.replace(',', ' ').replace('-', ' to '))
    self.changed = True