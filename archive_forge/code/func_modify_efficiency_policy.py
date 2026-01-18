from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_efficiency_policy(self, modify):
    """
        Modify a efficiency policy
        :return: None
        """
    if self.use_rest:
        return self.modify_efficiency_policy_rest(modify)
    sis_policy_obj = netapp_utils.zapi.NaElement('sis-policy-modify')
    sis_policy_obj.add_new_child('policy-name', self.parameters['policy_name'])
    for attribute in modify:
        sis_policy_obj.add_new_child(self.attribute_to_name(attribute), str(self.parameters[attribute]))
    try:
        self.server.invoke_successfully(sis_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying efficiency policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())