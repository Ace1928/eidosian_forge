from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_traffic_manager_endpoint(self):
    """
        Deletes the specified Traffic Manager endpoint.
        :return: True
        """
    self.log('Deleting the Traffic Manager endpoint {0}'.format(self.name))
    try:
        operation_result = self.traffic_manager_management_client.endpoints.delete(self.resource_group, self.profile_name, self.type, self.name)
        return True
    except Exception as exc:
        request_id = exc.request_id if exc.request_id else ''
        self.fail('Error deleting the Traffic Manager endpoint {0}, request id {1} - {2}'.format(self.name, request_id, str(exc)))
        return False