from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
def listbyservice(self):
    self.url = self.get_url_byservice()
    response = None
    try:
        response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        response = json.loads(response.body())
    except Exception as e:
        self.log('Could not get info for a given services.{0}'.format(e))
    try:
        response = json.loads(response.text)
    except Exception:
        return None
    return response