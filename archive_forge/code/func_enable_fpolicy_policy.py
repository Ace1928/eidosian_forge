from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def enable_fpolicy_policy(self):
    """
        Enables fPolicy policy
        :return: nothing
        """
    if self.use_rest:
        api = '/protocols/fpolicy/%s/policies/%s' % (self.svm_uuid, self.parameters['policy_name'])
        body = {'enabled': self.parameters['status'], 'priority': self.parameters['sequence_number']}
        dummy, error = self.rest_api.patch(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_enable_obj = netapp_utils.zapi.NaElement('fpolicy-enable-policy')
        fpolicy_enable_obj.add_new_child('policy-name', self.parameters['policy_name'])
        fpolicy_enable_obj.add_new_child('sequence-number', self.na_helper.get_value_for_int(False, self.parameters['sequence_number']))
        try:
            self.server.invoke_successfully(fpolicy_enable_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error enabling fPolicy policy %s on vserver %s: %s' % (self.parameters['policy_name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())