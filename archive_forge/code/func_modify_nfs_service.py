from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_nfs_service(self, modify):
    if self.use_rest:
        return self.modify_nfs_service_rest(modify)
    nfs_modify = netapp_utils.zapi.NaElement('nfs-service-modify')
    service_state = modify.pop('service_state', None)
    self.modify_service_state(service_state)
    for each in modify:
        if each in ['nfsv4_id_domain', 'tcp_max_xfer_size']:
            nfs_modify.add_new_child(self.zapi_names[each], str(modify[each]))
        else:
            nfs_modify.add_new_child(self.zapi_names[each], self.convert_to_bool(modify[each]))
    try:
        self.server.invoke_successfully(nfs_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying nfs: %s' % to_native(error), exception=traceback.format_exc())