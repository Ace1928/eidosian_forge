from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_functionapp(self):
    self.log('Get properties for Function App {0}'.format(self.name))
    function_app = None
    result = []
    try:
        function_app = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.name)
    except ResourceNotFoundError:
        pass
    if function_app and self.has_tags(function_app.tags, self.tags):
        result = function_app.as_dict()
    return [result]