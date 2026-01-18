from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_storage_share(self):
    """
        Method calling the Azure SDK to delete storage share.
        :return: object resulting from the original request
        """
    try:
        self.storage_client.file_shares.delete(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name)
    except Exception as e:
        self.fail('Error deleting file share {0} : {1}'.format(self.name, str(e)))
    return self.get_share()