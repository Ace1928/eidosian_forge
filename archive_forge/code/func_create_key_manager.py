from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_key_manager(self):
    """
        add key manager.
        """
    if self.use_rest:
        return self.create_key_manager_rest()
    key_manager_create = netapp_utils.zapi.NaElement('security-key-manager-add')
    key_manager_create.add_new_child('key-manager-ip-address', self.parameters['ip_address'])
    if self.parameters.get('tcp_port'):
        key_manager_create.add_new_child('key-manager-tcp-port', str(self.parameters['tcp_port']))
    try:
        self.cluster.invoke_successfully(key_manager_create, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating key manager: %s' % to_native(error), exception=traceback.format_exc())