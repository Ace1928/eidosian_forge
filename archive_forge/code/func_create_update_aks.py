from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_aks(self):
    """
        Creates or updates a managed Azure container service (AKS) with the specified configuration of agents.

        :return: deserialized AKS instance state dictionary
        """
    self.log('Creating / Updating the AKS instance {0}'.format(self.name))
    agentpools = []
    if self.agent_pool_profiles:
        agentpools = [self.create_agent_pool_profile_instance(profile) for profile in self.agent_pool_profiles]
    if self.service_principal:
        service_principal_profile = self.create_service_principal_profile_instance(self.service_principal)
        identity = None
    else:
        service_principal_profile = None
        identity = self.managedcluster_models.ManagedClusterIdentity(type='SystemAssigned')
    if self.linux_profile:
        linux_profile = self.create_linux_profile_instance(self.linux_profile)
    else:
        linux_profile = None
    parameters = self.managedcluster_models.ManagedCluster(location=self.location, dns_prefix=self.dns_prefix, kubernetes_version=self.kubernetes_version, tags=self.tags, service_principal_profile=service_principal_profile, agent_pool_profiles=agentpools, linux_profile=linux_profile, identity=identity, enable_rbac=self.enable_rbac, network_profile=self.create_network_profile_instance(self.network_profile), aad_profile=self.create_aad_profile_instance(self.aad_profile), api_server_access_profile=self.create_api_server_access_profile_instance(self.api_server_access_profile), addon_profiles=self.create_addon_profile_instance(self.addon), node_resource_group=self.node_resource_group)
    try:
        poller = self.managedcluster_client.managed_clusters.begin_create_or_update(self.resource_group, self.name, parameters)
        response = self.get_poller_result(poller)
        response.kube_config = self.get_aks_kubeconfig()
        return create_aks_dict(response)
    except Exception as exc:
        self.log('Error attempting to create the AKS instance.')
        self.fail('Error creating the AKS instance: {0}'.format(exc.message))