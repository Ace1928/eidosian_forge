from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import time
def delete_event_hub(self):
    """
        Deletes specified event hub
        :return True
        """
    self.log('Deleting the event hub {0}'.format(self.name))
    try:
        result = self.event_hub_client.event_hubs.delete(self.resource_group, self.namespace_name, self.name)
    except Exception as e:
        self.log('Error attempting to delete event hub.')
        self.fail('Error deleting the event hub : {0}'.format(str(e)))
    return True