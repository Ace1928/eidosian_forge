from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def lookup_datastore(self, confine_to_datacenter):
    """ Get datastore(s) per ESXi host or vCenter server """
    datastores = self.cache.get_all_objs(self.content, [vim.Datastore], confine_to_datacenter)
    return datastores