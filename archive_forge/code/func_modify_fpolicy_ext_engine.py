from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def modify_fpolicy_ext_engine(self, modify):
    """
        Modify an fPolicy external engine
        :return: nothing
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy/external-engine'
        query = {'vserver': self.parameters['vserver'], 'engine-name': self.parameters['name']}
        dummy, error = self.rest_api.patch(api, modify, query)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_ext_engine_obj = self.create_zapi_api('fpolicy-policy-external-engine-modify')
        try:
            self.server.invoke_successfully(fpolicy_ext_engine_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying fPolicy external engine %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())