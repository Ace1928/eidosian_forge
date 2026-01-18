from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_deleted_keys(self):
    """
        Lists deleted keys in specific key vault.

        :return: deserialized keys, includes key identifier, attributes and tags.
        """
    self.log('Get the key vaults in current subscription')
    results = []
    try:
        response = self._client.list_deleted_keys()
        self.log('Response : {0}'.format(response))
        if response:
            for item in response:
                item = deletedkeyitem_to_dict(item)
                if self.has_tags(item['tags'], self.tags):
                    results.append(item)
    except Exception as e:
        self.log('Did not find key vault in current subscription {0}.'.format(str(e)))
    return results