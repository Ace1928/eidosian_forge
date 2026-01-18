from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def add_kmip_to_standard_kp(self, kms_info, proxy_user_config_dict):
    kmip_server_info = self.create_kmip_server_info(kms_info, proxy_user_config_dict)
    kmip_server_spec = self.create_kmip_server_spec(self.key_provider_id, kmip_server_info, proxy_user_config_dict.get('kms_password'))
    try:
        self.crypto_mgr.RegisterKmipServer(server=kmip_server_spec)
    except Exception as e:
        self.module.fail_json(msg='Failed to add the KMIP server to Key Provider cluster with exception: %s' % to_native(e))