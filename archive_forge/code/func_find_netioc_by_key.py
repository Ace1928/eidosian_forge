from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def find_netioc_by_key(self, resource_name):
    config = None
    if self.dvs.config.networkResourceControlVersion == 'version3':
        config = self.dvs.config.infrastructureTrafficResourceConfig
    elif self.dvs.config.networkResourceControlVersion == 'version2':
        config = self.dvs.networkResourcePool
    for obj in config:
        if obj.key == resource_name:
            return obj
    return None