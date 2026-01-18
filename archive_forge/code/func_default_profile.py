from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def default_profile(self, cluster):
    """
        Used when upgrade profile returned by Azure in None
        (i.e. when the cluster runs latest version)
        :param: cluster: ManagedCluster with AKS instance information
        :return: dict containing upgrade profile with current cluster version
        """
    return dict(upgrades=None, kubernetes_version=cluster.kubernetes_version, name=None, os_type=None)