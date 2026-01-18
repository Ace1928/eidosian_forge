from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_policy_iter(self):
    policy_get_iter = netapp_utils.zapi.NaElement('file-directory-security-policy-get-iter')
    policy_info = netapp_utils.zapi.NaElement('file-directory-security-policy')
    policy_info.add_new_child('vserver', self.parameters['vserver'])
    policy_info.add_new_child('policy-name', self.parameters['policy_name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(policy_info)
    policy_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(policy_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching file-directory policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        policy = attributes_list.get_child_by_name('file-directory-security-policy')
        return policy.get_child_content('policy-name')
    return None