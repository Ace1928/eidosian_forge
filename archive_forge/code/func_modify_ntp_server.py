from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ntp_server(self, modify):
    """
        modify the ntp server
        """
    if self.use_rest:
        return self.modify_ntp_server_rest(modify)
    ntp_modify = netapp_utils.zapi.NaElement.create_node_with_children('ntp-server-modify', **{'server-name': self.parameters['server_name'], 'version': self.parameters['version']})
    if modify.get('key_id'):
        ntp_modify.add_new_child('key-id', str(self.parameters['key_id']))
    try:
        self.server.invoke_successfully(ntp_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying version for ntp server %s: %s' % (self.parameters['server_name'], to_native(error)), exception=traceback.format_exc())