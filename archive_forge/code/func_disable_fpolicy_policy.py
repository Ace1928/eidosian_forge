from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def disable_fpolicy_policy(self):
    """
        Disables fPolicy policy
        :return: nothing
        """
    if self.use_rest:
        api = '/protocols/fpolicy/%s/policies/%s' % (self.svm_uuid, self.parameters['policy_name'])
        body = {'enabled': self.parameters['status']}
        dummy, error = self.rest_api.patch(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_disable_obj = netapp_utils.zapi.NaElement('fpolicy-disable-policy')
        fpolicy_disable_obj.add_new_child('policy-name', self.parameters['policy_name'])
        try:
            self.server.invoke_successfully(fpolicy_disable_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error disabling fPolicy policy %s on vserver %s: %s' % (self.parameters['policy_name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())