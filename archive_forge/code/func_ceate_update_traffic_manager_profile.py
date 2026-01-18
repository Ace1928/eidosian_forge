from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def ceate_update_traffic_manager_profile(self):
    """
        Creates or updates a Traffic Manager profile.

        :return: deserialized Traffic Manager profile state dictionary
        """
    self.log('Creating / Updating the Traffic Manager profile {0}'.format(self.name))
    parameters = Profile(tags=self.tags, location=self.location, profile_status=self.profile_status, traffic_routing_method=self.traffic_routing_method, dns_config=create_dns_config_instance(self.dns_config) if self.dns_config else None, monitor_config=create_monitor_config_instance(self.monitor_config) if self.monitor_config else None, endpoints=create_endpoints(self.endpoints))
    try:
        response = self.traffic_manager_management_client.profiles.create_or_update(self.resource_group, self.name, parameters)
        return traffic_manager_profile_to_dict(response)
    except Exception as exc:
        self.log('Error attempting to create the Traffic Manager.')
        self.fail('Error creating the Traffic Manager: {0}'.format(exc.message))