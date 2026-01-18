from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def build_application_accesses(self, required_resource_accesses):
    if not required_resource_accesses:
        return None
    required_accesses = []
    if isinstance(required_resource_accesses, dict):
        self.log('Getting "requiredResourceAccess" from a full manifest')
        required_resource_accesses = required_resource_accesses.get('required_resource_access', [])
    for x in required_resource_accesses:
        accesses = [ResourceAccess(id=y['id'], type=y['type']) for y in x['resource_access']]
        required_accesses.append(RequiredResourceAccess(resource_app_id=x['resource_app_id'], resource_access=accesses))
    return required_accesses