from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def get_activedirectory_id(self):
    try:
        list_activedirectory, error = self.rest_api.get('Storage/ActiveDirectory')
    except Exception:
        return None
    if error is not None:
        self.module.fail_json(msg='Error calling list_activedirectory: %s' % error)
    for activedirectory in list_activedirectory:
        if activedirectory['region'] == self.parameters['region']:
            return activedirectory['UUID']
    return None