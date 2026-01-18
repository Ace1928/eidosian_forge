from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def format_endpoint_dict(self, name, key, endpoint, storagetype, protocol='https'):
    result = dict(endpoint=endpoint)
    if key:
        result['connectionstring'] = 'DefaultEndpointsProtocol={0};EndpointSuffix={1};AccountName={2};AccountKey={3};{4}Endpoint={5}'.format(protocol, self._cloud_environment.suffixes.storage_endpoint, name, key, str.title(storagetype), endpoint)
    return result