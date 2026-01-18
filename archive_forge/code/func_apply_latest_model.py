from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def apply_latest_model(self, instance_id):
    try:
        poller = self.compute_client.virtual_machine_scale_sets.begin_update_instances(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, vm_instance_i_ds={'instance_ids': instance_id})
        self.get_poller_result(poller)
    except Exception as exc:
        self.log('Error applying latest model {0} - {1}'.format(self.vmss_name, str(exc)))
        self.fail('Error applying latest model {0} - {1}'.format(self.vmss_name, str(exc)))