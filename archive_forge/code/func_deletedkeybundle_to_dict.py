from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def deletedkeybundle_to_dict(bundle):
    keybundle = delete_properties_to_dict(bundle.properties)
    keybundle['type'] = bundle.key_type
    keybundle['permitted_operations'] = bundle.key_operations
    keybundle['recovery_id'] = bundle.recovery_id
    keybundle['scheduled_purge_date'] = bundle.scheduled_purge_date
    keybundle['deleted_date'] = bundle.deleted_date
    keybundle['key'] = dict(n=bundle.key.n if hasattr(bundle.key, 'n') else None, e=bundle.key.e if hasattr(bundle.key, 'e') else None, crv=bundle.key.crv if hasattr(bundle.key, 'crv') else None, x=bundle.key.x if hasattr(bundle.key, 'x') else None, y=bundle.key.y if hasattr(bundle.key, 'y') else None)
    keybundle['id'] = bundle.id
    return keybundle