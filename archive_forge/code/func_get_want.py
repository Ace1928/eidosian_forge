from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_want(self):
    """
        Get all provision related informantion from the playbook
        Args:
            self: The instance of the class containing the 'config' attribute to be validated.
            config: validated config passed from the playbook
        Returns:
            The method returns an instance of the class with updated attributes:
                - self.want: A dictionary of paramters obtained from the playbook
                - self.msg: A message indicating all the paramters from the playbook are
                collected
                - self.status: Success
        Example:
            It stores all the paramters passed from the playbook for further processing
            before calling the APIs
        """
    self.want = {}
    self.want['device_type'] = self.get_dev_type()
    if self.want['device_type'] == 'wired':
        self.want['prov_params'] = self.get_wired_params()
    elif self.want['device_type'] == 'wireless':
        self.want['prov_params'] = self.get_wireless_params()
    else:
        self.log('Passed devices are neither wired or wireless devices', 'WARNING')
    self.msg = 'Successfully collected all parameters from playbook ' + 'for comparison'
    self.log(self.msg, 'INFO')
    self.status = 'success'
    return self