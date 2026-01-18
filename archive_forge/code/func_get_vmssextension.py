from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_vmssextension(self):
    self.log('Checking if the VMSS extension {0} is present'.format(self.name))
    try:
        response = self.compute_client.virtual_machine_scale_set_extensions.get(self.resource_group, self.vmss_name, self.name)
        return response.as_dict()
    except ResourceNotFoundError as e:
        self.log('Did not find VMSS extension')
        return False