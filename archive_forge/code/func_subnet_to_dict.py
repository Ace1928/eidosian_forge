from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def subnet_to_dict(self, subnet):
    result = dict(id=subnet.id, name=subnet.name, provisioning_state=subnet.provisioning_state, address_prefix=subnet.address_prefix, address_prefixes=subnet.address_prefixes if subnet.address_prefixes else None, network_security_group=subnet.network_security_group.id if subnet.network_security_group else None, route_table=subnet.route_table.id if subnet.route_table else None)
    if subnet.service_endpoints:
        result['service_endpoints'] = [{'service': item.service, 'locations': item.locations} for item in subnet.service_endpoints]
    return result