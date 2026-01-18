from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def modify_name_service_switch_rest(self):
    api = 'svm/svms'
    body = {'nsswitch': {self.parameters['database_type']: self.parameters['sources']}}
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
    if error:
        self.module.fail_json(msg='Error on modifying name service switch config on vserver %s: %s' % (self.parameters['vserver'], to_native(error)))