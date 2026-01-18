from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ems_filter(self, desired_rules):
    post_api = 'support/ems/filters/%s/rules' % self.parameters['name']
    api = 'support/ems/filters'
    if desired_rules['patch_rules'] != []:
        patch_body = {'rules': desired_rules['patch_rules']}
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['name'], patch_body)
        if error:
            self.module.fail_json(msg='Error modifying EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if desired_rules['post_rules'] != []:
        for rule in desired_rules['post_rules']:
            dummy, error = rest_generic.post_async(self.rest_api, post_api, rule)
            if error:
                self.module.fail_json(msg='Error modifying EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())