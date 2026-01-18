from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_dns_config_instance(dns_config):
    return DnsConfig(relative_name=dns_config['relative_name'], ttl=dns_config['ttl'])