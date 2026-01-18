from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_cluster_facts(self):
    cluster_facts = {'cluster': None}
    if self.host.parent and isinstance(self.host.parent, vim.ClusterComputeResource):
        cluster_facts.update(cluster=self.host.parent.name)
    return cluster_facts