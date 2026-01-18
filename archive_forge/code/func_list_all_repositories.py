from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_all_repositories(self, tag_name):
    response = None
    try:
        response = self._client.list_repository_names()
        self.log(f'Response : {response}')
    except Exception as e:
        self.fail(f'Could not get ACR repositories - {str(e)}')
    if response is not None:
        results = []
        for repo_name in response:
            tags = self.list_by_repository(repo_name, tag_name)
            if tags:
                results.append(tags)
        return results
    return None