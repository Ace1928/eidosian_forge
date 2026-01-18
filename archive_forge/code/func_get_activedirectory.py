from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def get_activedirectory(self, activedirectory_id=None):
    if activedirectory_id is None:
        return None
    else:
        activedirectory_info, error = self.rest_api.get('Storage/ActiveDirectory/%s' % activedirectory_id)
        if not error:
            return activedirectory_info
        return None