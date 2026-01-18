from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_network_details(self):
    """
            Check existing VLANs
            :return: vlan details if found, None otherwise
            :type: dict
        """
    vlans = self.elem.list_virtual_networks(virtual_network_tag=self.parameters['vlan_tag'])
    vlan_details = dict()
    for vlan in vlans.virtual_networks:
        if vlan is not None:
            vlan_details['name'] = vlan.name
            vlan_details['address_blocks'] = list()
            for address in vlan.address_blocks:
                vlan_details['address_blocks'].append({'start': address.start, 'size': address.size})
            vlan_details['svip'] = vlan.svip
            vlan_details['gateway'] = vlan.gateway
            vlan_details['netmask'] = vlan.netmask
            vlan_details['namespace'] = vlan.namespace
            vlan_details['attributes'] = vlan.attributes
            return vlan_details
    return None