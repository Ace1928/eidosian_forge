from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def search_cluster(self, datacenter_name, cluster_name, esxi_hostname):
    """
            Search cluster in vCenter
            Returns: host and cluster object
        """
    return find_host_by_cluster_datacenter(self.module, self.content, datacenter_name, cluster_name, esxi_hostname)