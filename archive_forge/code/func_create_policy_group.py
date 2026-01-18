from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def create_policy_group(self):
    """
        create a policy group name.
        """
    policy_group = netapp_utils.zapi.NaElement('qos-adaptive-policy-group-create')
    policy_group.add_new_child('policy-group', self.parameters['name'])
    policy_group.add_new_child('vserver', self.parameters['vserver'])
    if self.parameters.get('absolute_min_iops'):
        policy_group.add_new_child('absolute-min-iops', self.parameters['absolute_min_iops'])
    if self.parameters.get('expected_iops'):
        policy_group.add_new_child('expected-iops', self.parameters['expected_iops'])
    if self.parameters.get('peak_iops'):
        policy_group.add_new_child('peak-iops', self.parameters['peak_iops'])
    if self.parameters.get('peak_iops_allocation'):
        policy_group.add_new_child('peak-iops-allocation', self.parameters['peak_iops_allocation'])
    try:
        self.server.invoke_successfully(policy_group, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating adaptive qos policy group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())