from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_search(self):
    self.log('Creating search {0}'.format(self.name))
    self.check_values(self.hosting_mode, self.sku, self.partition_count, self.replica_count)
    self.check_name_availability()
    self.results['changed'] = True
    if self.network_rule_set:
        for rule in self.network_rule_set:
            self.firewall_list.append(self.search_client.services.models.IpRule(value=rule))
    search_model = self.search_client.services.models.SearchService(hosting_mode=self.hosting_mode, identity=self.search_client.services.models.Identity(type=self.identity), location=self.location, network_rule_set=dict(ip_rules=self.firewall_list) if len(self.firewall_list) > 0 else None, partition_count=self.partition_count, public_network_access=self.public_network_access, replica_count=self.replica_count, sku=self.search_client.services.models.Sku(name=self.sku), tags=self.tags)
    try:
        poller = self.search_client.services.begin_create_or_update(self.resource_group, self.name, search_model)
        self.get_poller_result(poller)
    except Exception as e:
        self.log('Error creating Azure Search.')
        self.fail('Failed to create Azure Search: {0}'.format(str(e)))
    return self.get_search()