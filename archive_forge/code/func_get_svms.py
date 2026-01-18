from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.um_info.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.um_info.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.um_info.plugins.module_utils.netapp import UMRestAPI
def get_svms(self):
    """
        Fetch details of svms.
        :return:
            Dictionary of current details if svms found
            None if svms is not found
        """
    data = {}
    api = 'datacenter/svm/svms'
    message, error = self.rest_api.get(api, data)
    if error:
        self.module.fail_json(msg=error)
    return self.rest_api.get_records(message, api)