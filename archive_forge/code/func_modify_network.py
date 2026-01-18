from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def modify_network(self, modify):
    """
            Modify the VLAN
        """
    try:
        self.elem.modify_virtual_network(virtual_network_tag=self.parameters['vlan_tag'], **modify)
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error modifying VLAN %s' % self.parameters['vlan_tag'], exception=to_native(err))