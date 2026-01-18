from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def invoke_snmp_community(self, zapi):
    """
        Invoke zapi - add/delete take the same NaElement structure
        """
    snmp_community = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **{'community': self.parameters['snmp_username'], 'access-control': self.parameters['access_control']})
    try:
        self.server.invoke_successfully(snmp_community, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if zapi == 'snmp-community-add':
            action = 'adding'
        elif zapi == 'snmp-community-delete':
            action = 'deleting'
        else:
            action = 'unexpected'
        self.module.fail_json(msg='Error %s community %s: %s' % (action, self.parameters['snmp_username'], to_native(error)), exception=traceback.format_exc())