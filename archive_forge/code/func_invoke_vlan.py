from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def invoke_vlan(self, zapi):
    """
        Invoke zapi - add/delete take the same NaElement structure
        """
    vlan_obj = netapp_utils.zapi.NaElement(zapi)
    vlan_info = self.create_vlan_info()
    vlan_obj.add_child_elem(vlan_info)
    try:
        self.server.invoke_successfully(vlan_obj, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if zapi == 'net-vlan-create':
            action = 'adding'
        elif zapi == 'net-vlan-delete':
            action = 'deleting'
        else:
            action = 'unexpected'
        self.module.fail_json(msg='Error %s Net Vlan %s: %s' % (action, self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())