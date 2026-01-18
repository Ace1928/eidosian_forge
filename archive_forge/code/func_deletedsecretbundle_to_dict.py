from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def deletedsecretbundle_to_dict(bundle):
    secretbundle = deleted_bundle_to_dict(bundle.properties)
    secretbundle['recovery_id'] = bundle._recovery_id
    secretbundle['scheduled_purge_date'] = bundle._scheduled_purge_date
    secretbundle['deleted_date'] = bundle._deleted_date
    return secretbundle