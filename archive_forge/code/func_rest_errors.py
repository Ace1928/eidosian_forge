from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def rest_errors(self):
    if self.parameters.get('qos_policy_group') and self.parameters.get('qos_adaptive_policy_group'):
        self.module.fail_json(msg='Error: With Rest API qos_policy_group and qos_adaptive_policy_group are now the same thing, and cannot be set at the same time')
    ontap_97_options = ['nas_application_template']
    if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 7) and any((x in self.parameters for x in ontap_97_options)):
        self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version(ontap_97_options, version='9.7'))
    if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9) and self.na_helper.safe_get(self.parameters, ['nas_application_template', 'flexcache', 'dr_cache']) is not None:
        self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version('flexcache: dr_cache', version='9.9'))
    if 'snapshot_auto_delete' in self.parameters:
        if 'destroy_list' in self.parameters['snapshot_auto_delete']:
            self.module.fail_json(msg="snapshot_auto_delete option 'destroy_list' is currently not supported with REST.")