from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_network_security_group_dict(nsg):
    result = dict(etag=nsg.etag, id=nsg.id, location=nsg.location, name=nsg.name, tags=nsg.tags, type=nsg.type)
    result['rules'] = []
    if nsg.security_rules:
        for rule in nsg.security_rules:
            result['rules'].append(create_rule_dict_from_obj(rule))
    result['default_rules'] = []
    if nsg.default_security_rules:
        for rule in nsg.default_security_rules:
            result['default_rules'].append(create_rule_dict_from_obj(rule))
    result['network_interfaces'] = []
    if nsg.network_interfaces:
        for interface in nsg.network_interfaces:
            result['network_interfaces'].append(interface.id)
    result['subnets'] = []
    if nsg.subnets:
        for subnet in nsg.subnets:
            result['subnets'].append(subnet.id)
    return result