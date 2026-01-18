from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
from ansible.module_utils._text import to_native
def gateway_backend_addr_pool_id(self, val):
    if isinstance(val, dict):
        appgw = val.get('application_gateway', None)
        name = val.get('name', None)
        if appgw and name:
            return resource_id(subscription=self.subscription_id, resource_group=self.resource_group, namespace='Microsoft.Network', type='applicationGateways', name=appgw, child_type_1='backendAddressPools', child_name_1=name)
    return val