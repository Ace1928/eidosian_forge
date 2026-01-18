from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_efficiency_policy(self):
    """
        Creates a efficiency policy
        :return: None
        """
    if self.use_rest:
        return self.create_efficiency_policy_rest()
    sis_policy_obj = netapp_utils.zapi.NaElement('sis-policy-create')
    for option, zapi_key in self.na_helper.zapi_int_keys.items():
        if self.parameters.get(option):
            sis_policy_obj.add_new_child(zapi_key, self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters[option]))
    for option, zapi_key in self.na_helper.zapi_bool_keys.items():
        if self.parameters.get(option):
            sis_policy_obj.add_new_child(zapi_key, self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters[option]))
    for option, zapi_key in self.na_helper.zapi_str_keys.items():
        if self.parameters.get(option):
            sis_policy_obj.add_new_child(zapi_key, str(self.parameters[option]))
    try:
        self.server.invoke_successfully(sis_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating efficiency policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())