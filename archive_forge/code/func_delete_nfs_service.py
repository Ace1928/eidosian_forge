from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_nfs_service(self):
    """
        delete nfs service.
        """
    if self.use_rest:
        return self.delete_nfs_service_rest()
    nfs_delete = netapp_utils.zapi.NaElement.create_node_with_children('nfs-service-destroy')
    try:
        self.server.invoke_successfully(nfs_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting nfs: %s' % to_native(error), exception=traceback.format_exc())