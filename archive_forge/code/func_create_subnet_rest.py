from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_subnet_rest(self):
    api = 'network/ip/subnets'
    dummy, error = rest_generic.post_async(self.rest_api, api, self.form_create_modify_body_rest())
    if error:
        self.module.fail_json(msg='Error creating subnet %s: %s' % (self.parameters['name'], to_native(error)))