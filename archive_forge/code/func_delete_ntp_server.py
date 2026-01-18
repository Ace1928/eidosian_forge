from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_ntp_server(self):
    """
        delete ntp server.
        """
    if self.use_rest:
        return self.delete_ntp_server_rest()
    ntp_server_delete = netapp_utils.zapi.NaElement.create_node_with_children('ntp-server-delete', **{'server-name': self.parameters['server_name']})
    try:
        self.server.invoke_successfully(ntp_server_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting ntp server %s: %s' % (self.parameters['server_name'], to_native(error)), exception=traceback.format_exc())