from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_modify_body(self, body, modify=None):
    params = modify or self.parameters
    if params.get('nfsv3') is not None:
        body['protocol.v3_enabled'] = self.convert_to_bool(params['nfsv3'])
    if params.get('nfsv4') is not None:
        body['protocol.v40_enabled'] = self.convert_to_bool(params['nfsv4'])
    if params.get('nfsv41') is not None:
        body['protocol.v41_enabled'] = self.convert_to_bool(params['nfsv41'])
    if params.get('nfsv41_pnfs') is not None:
        body['protocol.v41_features.pnfs_enabled'] = self.convert_to_bool(params['nfsv41_pnfs'])
    if params.get('vstorage_state') is not None:
        body['vstorage_enabled'] = self.convert_to_bool(params['vstorage_state'])
    if params.get('nfsv4_id_domain') is not None:
        body['protocol.v4_id_domain'] = params['nfsv4_id_domain']
    if params.get('tcp') is not None:
        body['transport.tcp_enabled'] = self.convert_to_bool(params['tcp'])
    if params.get('udp') is not None:
        body['transport.udp_enabled'] = self.convert_to_bool(params['udp'])
    if params.get('nfsv40_acl') is not None:
        body['protocol.v40_features.acl_enabled'] = self.convert_to_bool(params['nfsv40_acl'])
    if params.get('nfsv40_read_delegation') is not None:
        body['protocol.v40_features.read_delegation_enabled'] = self.convert_to_bool(params['nfsv40_read_delegation'])
    if params.get('nfsv40_write_delegation') is not None:
        body['protocol.v40_features.write_delegation_enabled'] = self.convert_to_bool(params['nfsv40_write_delegation'])
    if params.get('nfsv41_acl') is not None:
        body['protocol.v41_features.acl_enabled'] = self.convert_to_bool(params['nfsv41_acl'])
    if params.get('nfsv41_read_delegation') is not None:
        body['protocol.v41_features.read_delegation_enabled'] = self.convert_to_bool(params['nfsv41_read_delegation'])
    if params.get('nfsv41_write_delegation') is not None:
        body['protocol.v41_features.write_delegation_enabled'] = self.convert_to_bool(params['nfsv41_write_delegation'])
    if params.get('showmount') is not None:
        body['showmount_enabled'] = self.convert_to_bool(params['showmount'])
    if params.get('service_state') is not None:
        body['enabled'] = self.convert_to_bool(params['service_state'])
    if params.get('root') is not None:
        body['root'] = params['root']
    if params.get('windows') is not None:
        body['windows'] = params['windows']
    if params.get('security') is not None:
        body['security'] = params['security']
    if params.get('tcp_max_xfer_size') is not None:
        body['transport.tcp_max_transfer_size'] = params['tcp_max_xfer_size']
    return body