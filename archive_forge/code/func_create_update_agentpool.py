from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_agentpool(self, to_update_name_list):
    response_all = []
    for profile in self.agent_pool_profiles:
        if profile['name'] in to_update_name_list:
            self.log('Creating / Updating the AKS agentpool {0}'.format(profile['name']))
            parameters = self.managedcluster_models.AgentPool(count=profile['count'], vm_size=profile['vm_size'], os_disk_size_gb=profile['os_disk_size_gb'], max_count=profile['max_count'], node_labels=profile['node_labels'], min_count=profile['min_count'], orchestrator_version=profile['orchestrator_version'], max_pods=profile['max_pods'], enable_auto_scaling=profile['enable_auto_scaling'], agent_pool_type=profile['type'], mode=profile['mode'])
            try:
                poller = self.managedcluster_client.agent_pools.begin_create_or_update(self.resource_group, self.name, profile['name'], parameters)
                response = self.get_poller_result(poller)
                response_all.append(response)
            except Exception as exc:
                self.fail('Error attempting to update AKS agentpool: {0}'.format(exc.message))
    return create_agent_pool_profiles_dict(response_all)