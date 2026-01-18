from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def create_kmip_server_spec(key_provider_id, kms_server_info, kms_password=None):
    kmip_server_spec = None
    if key_provider_id and kms_server_info:
        kmip_server_spec = vim.encryption.KmipServerSpec()
        kmip_server_spec.clusterId = key_provider_id
        kmip_server_spec.info = kms_server_info
        if kms_password:
            kmip_server_spec.password = kms_password
    return kmip_server_spec