from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_resource_pool_by_name(self, datacenter_name, resourcepool_name, cluster_name=None, host_name=None):
    """
        Returns the identifier of a resource pool
        with the mentioned names.
        """
    datacenter = self.get_datacenter_by_name(datacenter_name)
    if not datacenter:
        return None
    clusters = None
    if cluster_name:
        clusters = self.get_cluster_by_name(datacenter_name, cluster_name)
        if clusters:
            clusters = set([clusters])
    hosts = None
    if host_name:
        hosts = self.get_host_by_name(datacenter_name, host_name)
        if hosts:
            hosts = set([hosts])
    names = set([resourcepool_name]) if resourcepool_name else None
    filter_spec = ResourcePool.FilterSpec(datacenters=set([datacenter]), names=names, clusters=clusters)
    resource_pool_summaries = self.api_client.vcenter.ResourcePool.list(filter_spec)
    resource_pool = resource_pool_summaries[0].resource_pool if len(resource_pool_summaries) > 0 else None
    return resource_pool