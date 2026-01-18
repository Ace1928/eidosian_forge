from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_key_manager(self):
    """
        delete key manager.
        """
    if self.use_rest:
        return self.delete_key_manager_rest()
    key_manager_delete = netapp_utils.zapi.NaElement('security-key-manager-delete')
    key_manager_delete.add_new_child('key-manager-ip-address', self.parameters['ip_address'])
    try:
        self.cluster.invoke_successfully(key_manager_delete, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting key manager: %s' % to_native(error), exception=traceback.format_exc())