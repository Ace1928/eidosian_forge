from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
def delete_diskencryptionset(self):
    try:
        response = self.compute_client.disk_encryption_sets.begin_delete(resource_group_name=self.resource_group, disk_encryption_set_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error deleting disk encryption set {0} - {1}'.format(self.name, str(exc)))
    return response