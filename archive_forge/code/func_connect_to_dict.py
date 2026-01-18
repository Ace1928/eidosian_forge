from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def connect_to_dict(self, connect_info):
    connect = connect_info.as_dict()
    result = dict(id=connect.get('id'), name=connect.get('name'), type=connect.get('type'), etag=connect.get('etag'), private_endpoint=dict(), private_link_service_connection_state=dict(), provisioning_state=connect.get('provisioning_state'), link_identifier=connect.get('link_identifier'))
    if connect.get('private_endpoint') is not None:
        result['private_endpoint']['id'] = connect.get('private_endpoint')['id']
    if connect.get('private_link_service_connection_state') is not None:
        result['private_link_service_connection_state']['status'] = connect.get('private_link_service_connection_state')['status']
        result['private_link_service_connection_state']['description'] = connect.get('private_link_service_connection_state')['description']
        result['private_link_service_connection_state']['actions_required'] = connect.get('private_link_service_connection_state')['actions_required']
    return result