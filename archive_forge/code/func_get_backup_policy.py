from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_backup_policy(self):
    """
        Gets the properties of the specified backup policy.

        :return: ProtectionPolicyResource
        """
    self.log('Checking if the backup policy {0} for vault {1} in resource group {2} is present'.format(self.name, self.vault_name, self.resource_group))
    policy = None
    try:
        policy = self.recovery_services_backup_client.protection_policies.get(vault_name=self.vault_name, resource_group_name=self.resource_group, policy_name=self.name)
    except ResourceNotFoundError as ex:
        self.log('Could not find backup policy {0} for vault {1} in resource group {2}'.format(self.name, self.vault_name, self.resource_group))
    return policy