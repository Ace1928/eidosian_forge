from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_nfs_service_rest(self):
    api = 'protocols/nfs/services'
    params = {'svm.name': self.parameters['vserver'], 'fields': 'protocol.v3_enabled,protocol.v40_enabled,protocol.v41_enabled,protocol.v41_features.pnfs_enabled,vstorage_enabled,protocol.v4_id_domain,transport.tcp_enabled,transport.udp_enabled,protocol.v40_features.acl_enabled,protocol.v40_features.read_delegation_enabled,protocol.v40_features.write_delegation_enabled,protocol.v41_features.acl_enabled,protocol.v41_features.read_delegation_enabled,protocol.v41_features.write_delegation_enabled,enabled,svm.uuid,'}
    if self.parameters.get('showmount'):
        params['fields'] += 'showmount_enabled,'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 0):
        params['fields'] += 'root.*,security.*,windows.*,transport.tcp_max_transfer_size'
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error getting nfs services for SVM %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 0):
        if record and 'default_user' not in record.get('windows'):
            record['windows']['default_user'] = None
    return self.format_get_nfs_service_rest(record) if record else record