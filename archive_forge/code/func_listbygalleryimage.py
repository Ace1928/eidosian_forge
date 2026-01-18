from __future__ import absolute_import, division, print_function
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
def listbygalleryimage(self):
    response = None
    results = dict(response=[])
    self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.Compute' + '/galleries' + '/{{ gallery_name }}' + '/images' + '/{{ image_name }}' + '/versions'
    self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
    self.url = self.url.replace('{{ resource_group }}', self.resource_group)
    self.url = self.url.replace('{{ gallery_name }}', self.gallery_name)
    self.url = self.url.replace('{{ image_name }}', self.gallery_image_name)
    try:
        skiptoken = None
        while True:
            if skiptoken:
                self.query_parameters['skiptoken'] = skiptoken
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, [200, 404], 0, 0)
            try:
                response = json.loads(response.body())
                if isinstance(response, dict):
                    if response.get('value'):
                        results['response'] = results['response'] + response['value']
                        skiptoken = response.get('nextLink')
                    else:
                        results['response'] = results['response'] + [response]
            except Exception as e:
                self.fail('Failed to parse response: ' + str(e))
            if not skiptoken:
                break
    except Exception as e:
        self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
    return [self.format_item(x) for x in results['response']] if results['response'] else []