from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def get_source_vm(self):
    resource = dict()
    if isinstance(self.source, dict):
        if self.source.get('type') != 'virtual_machines':
            return None
        resource = dict(type='virtualMachines', name=self.source['name'], resource_group=self.source.get('resource_group') or self.resource_group)
    elif isinstance(self.source, str):
        vm_resource_id = format_resource_id(self.source, self.subscription_id, 'Microsoft.Compute', 'virtualMachines', self.resource_group)
        resource = parse_resource_id(vm_resource_id)
    else:
        self.fail('Unsupported type of source parameter, please give string or dictionary')
    return self.get_vm(resource['resource_group'], resource['name']) if resource['type'] == 'virtualMachines' else None