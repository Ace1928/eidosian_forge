from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_cdnendpoint(self):
    """
        Deletes the specified Azure CDN endpoint in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Azure CDN endpoint {0}'.format(self.name))
    try:
        poller = self.cdn_client.endpoints.begin_delete(self.resource_group, self.profile_name, self.name)
        self.get_poller_result(poller)
        return True
    except Exception as e:
        self.log('Error attempting to delete the Azure CDN endpoint.')
        self.fail('Error deleting the Azure CDN endpoint: {0}'.format(e.message))
        return False