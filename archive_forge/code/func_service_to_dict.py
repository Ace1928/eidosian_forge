from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def service_to_dict(self, service_info):
    service = service_info.as_dict()
    result = dict(id=service.get('id'), name=service.get('name'), type=service.get('type'), etag=service.get('etag'), location=service.get('location'), tags=service.get('tags'), load_balancer_frontend_ip_configurations=service.get('load_balancer_frontend_ip_configurations'), ip_configurations=list(), network_interfaces=service.get('network_interfaces'), provisioning_state=service.get('provisioning_state'), private_endpoint_connections=list(), visibility=service.get('visibility'), auto_approval=service.get('auto_approval'), fqdns=service.get('fqdns'), alias=service.get('alias'), enable_proxy_protocol=service.get('enable_proxy_protocol'))
    if service.get('private_endpoint_connections'):
        for items in service['private_endpoint_connections']:
            result['private_endpoint_connections'].append({'id': items['id'], 'private_endpoint': items['private_endpoint']['id']})
    if service.get('ip_configurations'):
        for items in service['ip_configurations']:
            result['ip_configurations'].append({'name': items['name'], 'properties': {'primary': items['primary'], 'private_ip_address_version': items['private_ip_address_version'], 'private_ip_allocation_method': items['private_ip_allocation_method'], 'subnet': items['subnet']}})
    return result