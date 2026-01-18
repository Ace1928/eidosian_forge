from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def get_key_provider_clusters(self):
    key_provider_clusters = None
    try:
        if self.vcenter_version_at_least(version=(7, 0, 0)):
            key_provider_clusters = self.crypto_mgr.ListKmsClusters(includeKmsServers=True)
        else:
            key_provider_clusters = self.crypto_mgr.ListKmipServers()
    except Exception as e:
        self.module.fail_json(msg='Failed to get key provider clusters info with exception: %s' % to_native(e))
    return key_provider_clusters