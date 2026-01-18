from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def get_resource_pool(self, cluster=None, host=None, resource_pool=None):
    """ Get a resource pool, filter on cluster, esxi_hostname or resource_pool if given """
    cluster_name = cluster or self.params.get('cluster', None)
    host_name = host or self.params.get('esxi_hostname', None)
    resource_pool_name = resource_pool or self.params.get('resource_pool', None)
    datacenter = find_obj(self.content, [vim.Datacenter], self.params['datacenter'])
    if not datacenter:
        self.module.fail_json(msg='Unable to find datacenter "%s"' % self.params['datacenter'])
    if cluster_name:
        cluster = find_obj(self.content, [vim.ComputeResource], cluster_name, folder=datacenter)
        if not cluster:
            self.module.fail_json(msg='Unable to find cluster "%s"' % cluster_name)
    elif host_name:
        host = find_obj(self.content, [vim.HostSystem], host_name, folder=datacenter)
        if not host:
            self.module.fail_json(msg='Unable to find host "%s"' % host_name)
        cluster = host.parent
    else:
        cluster = None
    resource_pool = find_obj(self.content, [vim.ResourcePool], resource_pool_name, folder=cluster or datacenter)
    if not resource_pool:
        if resource_pool_name:
            self.module.fail_json(msg='Unable to find resource_pool "%s"' % resource_pool_name)
        else:
            self.module.fail_json(msg='Unable to find resource pool, need esxi_hostname, resource_pool, or cluster')
    return resource_pool