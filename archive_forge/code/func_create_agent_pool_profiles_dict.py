from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_agent_pool_profiles_dict(agentpoolprofiles):
    """
    Helper method to deserialize a ContainerServiceAgentPoolProfile to a dict
    :param: agentpoolprofiles: ContainerServiceAgentPoolProfile with the Azure callback object
    :return: dict with the state on Azure
    """
    return [dict(count=profile.count, vm_size=profile.vm_size, name=profile.name, os_disk_size_gb=profile.os_disk_size_gb, vnet_subnet_id=profile.vnet_subnet_id, availability_zones=profile.availability_zones, os_type=profile.os_type, type=profile.type, mode=profile.mode, orchestrator_version=profile.orchestrator_version, enable_auto_scaling=profile.enable_auto_scaling, max_count=profile.max_count, node_labels=profile.node_labels, min_count=profile.min_count, max_pods=profile.max_pods) for profile in agentpoolprofiles] if agentpoolprofiles else None