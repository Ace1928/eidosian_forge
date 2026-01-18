from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def delete_fpolicy_policy(self):
    """
        Delete an FPolicy policy.
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy'
        body = {'vserver': self.parameters['vserver'], 'policy-name': self.parameters['name']}
        dummy, error = self.rest_api.delete(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_policy_obj = netapp_utils.zapi.NaElement('fpolicy-policy-delete')
        fpolicy_policy_obj.add_new_child('policy-name', self.parameters['name'])
        try:
            self.server.invoke_successfully(fpolicy_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting fPolicy policy %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())