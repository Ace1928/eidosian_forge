from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_properties_to_dict(bundle):
    return dict(tags=bundle.tags, managed=bundle.managed, attributes=dict(enabled=bundle.enabled, not_before=bundle.not_before, expires=bundle.expires_on, created=bundle.created_on, updated=bundle.updated_on, recovery_level=bundle.recovery_level), kid=bundle.id, version=bundle.version)