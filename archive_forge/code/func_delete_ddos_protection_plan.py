from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_ddos_protection_plan(self):
    """
        Deletes specified DDoS protection plan
        :return True
        """
    self.log('Deleting the DDoS protection plan {0}'.format(self.name))
    try:
        poller = self.network_client.ddos_protection_plans.begin_delete(self.resource_group, self.name)
        result = self.get_poller_result(poller)
    except ResourceNotFoundError as e:
        self.log('Error attempting to delete DDoS protection plan.')
        self.fail('Error deleting the DDoS protection plan : {0}'.format(str(e)))
    return result