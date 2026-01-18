from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_nfs_service(self):
    if self.use_rest:
        return self.get_nfs_service_rest()
    nfs_get_iter = netapp_utils.zapi.NaElement('nfs-service-get-iter')
    nfs_info = netapp_utils.zapi.NaElement('nfs-info')
    nfs_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(nfs_info)
    nfs_get_iter.add_child_elem(query)
    result = self.server.invoke_successfully(nfs_get_iter, True)
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        return self.format_return(result)
    return None