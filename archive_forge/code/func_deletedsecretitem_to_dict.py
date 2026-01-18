from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def deletedsecretitem_to_dict(secretitem):
    item = secretitem_to_dict(secretitem.properties)
    item['recovery_id'] = secretitem._recovery_id
    item['scheduled_purge_date'] = secretitem._scheduled_purge_date
    item['deleted_date'] = secretitem._deleted_date
    return item