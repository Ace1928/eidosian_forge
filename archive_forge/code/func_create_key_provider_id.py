from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def create_key_provider_id(key_provider_name):
    key_provider_id = None
    if key_provider_name:
        key_provider_id = vim.encryption.KeyProviderId()
        key_provider_id.id = key_provider_name
    return key_provider_id