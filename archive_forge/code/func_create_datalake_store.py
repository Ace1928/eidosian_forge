from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_datalake_store(self):
    self.log('Creating datalake store {0}'.format(self.name))
    if not self.location:
        self.fail('Parameter error: location required when creating a datalake store account.')
    self.check_name_availability()
    self.results['changed'] = True
    if self.check_mode:
        account_dict = dict(name=self.name, resource_group=self.resource_group, location=self.location)
        return account_dict
    if self.firewall_rules is not None:
        self.firewall_rules_model = list()
        for rule in self.firewall_rules:
            rule_model = self.datalake_store_models.CreateFirewallRuleWithAccountParameters(name=rule.get('name'), start_ip_address=rule.get('start_ip_address'), end_ip_address=rule.get('end_ip_address'))
            self.firewall_rules_model.append(rule_model)
    if self.virtual_network_rules is not None:
        self.virtual_network_rules_model = list()
        for vnet_rule in self.virtual_network_rules:
            vnet_rule_model = self.datalake_store_models.CreateVirtualNetworkRuleWithAccountParameters(name=vnet_rule.get('name'), subnet_id=vnet_rule.get('subnet_id'))
            self.virtual_network_rules_model.append(vnet_rule_model)
    parameters = self.datalake_store_models.CreateDataLakeStoreAccountParameters(default_group=self.default_group, encryption_config=self.encryption_config_model, encryption_state=self.encryption_state, firewall_allow_azure_ips=self.firewall_allow_azure_ips, firewall_rules=self.firewall_rules_model, firewall_state=self.firewall_state, identity=self.identity_model, location=self.location, new_tier=self.new_tier, tags=self.tags, virtual_network_rules=self.virtual_network_rules_model)
    self.log(str(parameters))
    try:
        poller = self.datalake_store_client.accounts.begin_create(self.resource_group, self.name, parameters)
        self.get_poller_result(poller)
    except Exception as e:
        self.log('Error creating datalake store.')
        self.fail('Failed to create datalake store: {0}'.format(str(e)))
    return self.get_datalake_store()