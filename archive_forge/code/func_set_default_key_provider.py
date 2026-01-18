from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def set_default_key_provider(self):
    try:
        self.crypto_mgr.MarkDefault(self.key_provider_id)
    except Exception as e:
        self.module.fail_json(msg="Failed to mark default key provider to '%s' with exception: %s" % (self.key_provider_id.id, to_native(e)))