from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def ddos_protection_plan_to_dict(self, item):
    ddos_protection_plan = item.as_dict()
    result = dict(additional_properties=ddos_protection_plan.get('additional_properties', None), id=ddos_protection_plan.get('id', None), name=ddos_protection_plan.get('name', None), type=ddos_protection_plan.get('type', None), location=ddos_protection_plan.get('location', None), tags=ddos_protection_plan.get('tags', None), etag=ddos_protection_plan.get('etag', None), resource_guid=ddos_protection_plan.get('resource_guid', None), provisioning_state=ddos_protection_plan.get('provisioning_state', None), virtual_networks=ddos_protection_plan.get('virtual_networks', None))
    return result