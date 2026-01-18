from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def create_discovery(self, ip_address_list=None):
    """
        Start a new discovery process in the Cisco Catalyst Center. It creates the
        parameters required for the discovery and then calls the
        'start_discovery' function. The result of the discovery process
        is added to the 'result' attribute.

        Parameters:
          - credential_ids: The list of credential IDs to include in the
                            discovery. If not provided, an empty list is used.
          - ip_address_list: The list of IP addresses to include in the
                             discovery. If not provided, None is used.

        Returns:
          - task_id: The ID of the task created for the discovery process.
        """
    result = self.dnac_apply['exec'](family='discovery', function='start_discovery', params=self.create_params(ip_address_list=ip_address_list), op_modifies=True)
    self.log('The response received post discovery creation API called is {0}'.format(str(result)), 'DEBUG')
    self.result.update(dict(discovery_result=result))
    self.log('Task Id of the API task created is {0}'.format(result.response.get('taskId')), 'INFO')
    return result.response.get('taskId')