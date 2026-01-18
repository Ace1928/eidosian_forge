from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_notification_hub(self):
    """
        Create or update Notification Hub.
        :return: create or update Notification Hub instance state dictionary
        """
    try:
        response = self.create_or_update_namespaces()
        params = NotificationHubCreateOrUpdateParameters(location=self.location, sku=Sku(name=self.sku), tags=self.tags)
        result = self.notification_hub_client.notification_hubs.create_or_update(self.resource_group, self.namespace_name, self.name, params)
        self.log('Response : {0}'.format(result))
    except Exception as ex:
        self.fail('Failed to create notification hub {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
    return notification_hub_to_dict(result)