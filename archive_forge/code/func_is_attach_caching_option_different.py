from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_attach_caching_option_different(self, vm_name, disk):
    resp = False
    if vm_name:
        vm = self._get_vm(self.resource_group, vm_name)
        correspondence = next((d for d in vm.storage_profile.data_disks if d.name.lower() == disk.get('name').lower()), None)
        caching_options = self.compute_models.CachingTypes[self.attach_caching] if self.attach_caching and self.attach_caching != '' else None
        if correspondence and correspondence.caching != caching_options:
            resp = True
            if correspondence.caching == 'none' and (self.attach_caching == '' or self.attach_caching is None):
                resp = False
    return resp