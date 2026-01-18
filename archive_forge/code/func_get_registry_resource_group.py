from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def get_registry_resource_group(self, registry_name):
    response = None
    try:
        response = self.containerregistry_client.registries.list()
    except Exception as e:
        self.fail(f'Could not load resource group for registry {registry_name} - {str(e)}')
    if response is not None:
        for item in response:
            item_dict = item.as_dict()
            if item_dict['name'] == registry_name:
                return azure_id_to_dict(item_dict['id']).get('resourceGroups')
    return None