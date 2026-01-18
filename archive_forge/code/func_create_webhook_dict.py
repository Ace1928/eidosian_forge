from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_webhook_dict(webhook):
    if webhook is None:
        return None
    results = dict(id=webhook.id, name=webhook.name, location=webhook.location, provisioning_state=webhook.provisioning_state, tags=webhook.tags, status=webhook.status)
    return results