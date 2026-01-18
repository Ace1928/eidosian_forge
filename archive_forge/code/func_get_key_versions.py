from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_key_versions(self):
    """
        Lists keys versions.

        :return: deserialized versions of key, includes key identifier, attributes and tags
        """
    self.log('Get the key versions {0}'.format(self.name))
    results = []
    try:
        response = self._client.list_properties_of_key_versions(name=self.name)
        self.log('Response : {0}'.format(response))
        if response:
            for item in response:
                item = keyitem_to_dict(item)
                if self.has_tags(item['tags'], self.tags):
                    results.append(item)
    except Exception as e:
        self.log('Did not find key versions {0} : {1}.'.format(self.name, str(e)))
    return results