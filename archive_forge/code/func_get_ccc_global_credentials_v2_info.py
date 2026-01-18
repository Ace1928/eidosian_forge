from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_ccc_global_credentials_v2_info(self):
    """
        Retrieve the global credentials information (version 2).
        It applies the 'get_all_global_credentials_v2' function and extracts
        the IDs of the credentials. If no credentials are found, the
        function fails with a message.

        Returns:
          This method does not return a value. However, updates the attributes:
          - self.creds_ids_list: The list of credentials IDs is extended with
                                 the IDs extracted from the response.
          - self.result: A dictionary that is updated with the credentials IDs.
        """
    response = self.dnac_apply['exec'](family='discovery', function='get_all_global_credentials_v2', params=self.validated_config[0].get('headers'))
    response = response.get('response')
    self.log("The Global credentials response from 'get all global credentials v2' API is {0}".format(str(response)), 'DEBUG')
    global_credentials_all = {}
    global_credentials = self.validated_config[0].get('global_credentials')
    if global_credentials:
        global_credentials_all = self.handle_global_credentials(response=response)
    global_cred_set = set(global_credentials_all.keys())
    response_cred_set = set(response.keys())
    diff_keys = response_cred_set.difference(global_cred_set)
    for key in diff_keys:
        global_credentials_all[key] = []
        if response[key] is None:
            response[key] = []
        total_len = len(response[key])
        if total_len > 5:
            total_len = 5
        for element in response.get(key):
            global_credentials_all[key].append(element.get('id'))
        global_credentials_all[key] = global_credentials_all[key][:total_len]
    if global_credentials_all == {}:
        msg = 'Not found any global credentials to perform discovery'
        self.log(msg, 'WARNING')
    return global_credentials_all