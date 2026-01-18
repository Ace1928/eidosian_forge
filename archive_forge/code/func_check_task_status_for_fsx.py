from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def check_task_status_for_fsx(self, api_url):
    network_retries = 3
    exponential_retry_time = 1
    while True:
        result, error, dummy = self.rest_api.get(api_url, None, header=self.headers)
        if error is not None:
            if network_retries > 0:
                time.sleep(exponential_retry_time)
                exponential_retry_time *= 2
                network_retries = network_retries - 1
            else:
                return (0, error)
        else:
            response = result
            break
    return (response['providerDetails'], None)