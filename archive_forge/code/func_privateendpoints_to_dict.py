from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def privateendpoints_to_dict(self, privateendpoint):
    if privateendpoint is None:
        return None
    results = dict(id=privateendpoint.id, name=privateendpoint.name, location=privateendpoint.location, tags=privateendpoint.tags, provisioning_state=privateendpoint.provisioning_state, type=privateendpoint.type, etag=privateendpoint.etag, subnet_id=privateendpoint.subnet.id)
    if privateendpoint.network_interfaces and len(privateendpoint.network_interfaces) > 0:
        results['network_interfaces'] = []
        for interface in privateendpoint.network_interfaces:
            results['network_interfaces'].append(interface.id)
    if privateendpoint.private_link_service_connections and len(privateendpoint.private_link_service_connections) > 0:
        results['private_link_service_connections'] = []
        for connections in privateendpoint.private_link_service_connections:
            connection = {}
            connection['connection_state'] = {}
            connection['id'] = connections.id
            connection['name'] = connections.name
            connection['type'] = connections.type
            connection['group_ids'] = connections.group_ids
            connection['connection_state']['status'] = connections.private_link_service_connection_state.status
            connection['connection_state']['description'] = connections.private_link_service_connection_state.description
            connection['connection_state']['actions_required'] = connections.private_link_service_connection_state.actions_required
            results['private_link_service_connections'].append(connection)
    if privateendpoint.manual_private_link_service_connections and len(privateendpoint.manual_private_link_service_connections) > 0:
        results['manual_private_link_service_connections'] = []
        for connections in privateendpoint.manual_private_link_service_connections:
            results['manual_private_link_service_connections'].append(connections.id)
    return results