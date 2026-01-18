from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def deleted_bundle_to_dict(bundle):
    return dict(tags=bundle.tags, attributes=dict(enabled=bundle.enabled, not_before=bundle.not_before, expires=bundle.expires_on, created=bundle.created_on, updated=bundle.updated_on, recovery_level=bundle.recovery_level), sid=bundle.id, version=bundle.key_id, content_type=bundle.content_type, secret=bundle.version)