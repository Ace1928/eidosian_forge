from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_policy_group_rest(self, modify):
    api = 'storage/qos/policies'
    body = {}
    if 'fixed_qos_options' in modify:
        body['fixed'] = modify['fixed_qos_options']
    else:
        if 'block_size' not in self.na_helper.safe_get(modify, ['adaptive_qos_options']) and self.na_helper.safe_get(self.parameters, ['adaptive_qos_options', 'block_size']) is None:
            del self.parameters['adaptive_qos_options']['block_size']
        body['adaptive'] = self.parameters['adaptive_qos_options']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying qos policy group %s: %s' % (self.parameters['name'], error))