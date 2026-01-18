from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_webhook(self):
    """
        Gets the properties of the specified Webhook.

        :return: deserialized Webhook instance state dictionary
        """
    self.log('Checking if the Webhook instance {0} is present'.format(self.webhook_name))
    found = False
    try:
        response = self.containerregistry_client.webhooks.get(resource_group_name=self.resource_group, registry_name=self.registry_name, webhook_name=self.webhook_name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('Webhook instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the Webhook instance: {0}'.format(str(e)))
    if found is True:
        return response.as_dict()
    return False