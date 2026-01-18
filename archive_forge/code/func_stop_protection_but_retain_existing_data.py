from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
def stop_protection_but_retain_existing_data(self):
    try:
        response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
    except Exception as e:
        self.log('Error attempting to stop protection.')
        self.fail('Error in disabling the protection: {0}'.format(str(e)))
    try:
        response = json.loads(response.body())
    except Exception:
        response = {'text': response.context['deserialized_data']}
    return response