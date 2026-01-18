from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def delete_policy_group(self, policy_group=None):
    """
        delete an existing policy group.
        :param policy_group: policy group name.
        """
    if policy_group is None:
        policy_group = self.parameters['name']
    policy_group_obj = netapp_utils.zapi.NaElement('qos-adaptive-policy-group-delete')
    policy_group_obj.add_new_child('policy-group', policy_group)
    if self.parameters.get('force'):
        policy_group_obj.add_new_child('force', str(self.parameters['force']))
    try:
        self.server.invoke_successfully(policy_group_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting adaptive qos policy group %s: %s' % (policy_group, to_native(error)), exception=traceback.format_exc())