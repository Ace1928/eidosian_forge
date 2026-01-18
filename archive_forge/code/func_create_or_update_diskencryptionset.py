from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
def create_or_update_diskencryptionset(self, disk_encryption_set):
    try:
        response = self.compute_client.disk_encryption_sets.begin_create_or_update(resource_group_name=self.resource_group, disk_encryption_set_name=self.name, disk_encryption_set=disk_encryption_set)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error creating or updating disk encryption set {0} - {1}'.format(self.name, str(exc)))
    return self.diskencryptionset_to_dict(response)