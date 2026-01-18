from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_ems_filter(self):
    api = 'support/ems/filters'
    body = {'name': self.parameters['name']}
    if self.parameters.get('rules'):
        body['rules'] = self.na_helper.filter_out_none_entries(self.parameters['rules'])
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())