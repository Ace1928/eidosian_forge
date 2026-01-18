from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def delete_fpolicy_scope(self):
    """
        Delete an FPolicy policy scope
        :return: nothing
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy/scope'
        body = {'vserver': self.parameters['vserver'], 'policy-name': self.parameters['name']}
        dummy, error = self.rest_api.delete(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_scope_obj = netapp_utils.zapi.NaElement('fpolicy-policy-scope-delete')
        fpolicy_scope_obj.add_new_child('policy-name', self.parameters['name'])
        try:
            self.server.invoke_successfully(fpolicy_scope_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting fPolicy policy scope %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())