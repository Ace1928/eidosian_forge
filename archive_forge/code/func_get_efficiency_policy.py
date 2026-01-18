from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_efficiency_policy(self):
    """
        Get a efficiency policy
        :return: a efficiency-policy info
        """
    if self.use_rest:
        return self.get_efficiency_policy_rest()
    sis_policy_obj = netapp_utils.zapi.NaElement('sis-policy-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    sis_policy_info = netapp_utils.zapi.NaElement('sis-policy-info')
    sis_policy_info.add_new_child('policy-name', self.parameters['policy_name'])
    sis_policy_info.add_new_child('vserver', self.parameters['vserver'])
    query.add_child_elem(sis_policy_info)
    sis_policy_obj.add_child_elem(query)
    try:
        results = self.server.invoke_successfully(sis_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error searching for efficiency policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())
    return_value = {}
    if results.get_child_by_name('num-records') and int(results.get_child_content('num-records')) == 1:
        attributes_list = results.get_child_by_name('attributes-list')
        sis_info = attributes_list.get_child_by_name('sis-policy-info')
        for option, zapi_key in self.na_helper.zapi_int_keys.items():
            return_value[option] = self.na_helper.get_value_for_int(from_zapi=True, value=sis_info.get_child_content(zapi_key))
        for option, zapi_key in self.na_helper.zapi_bool_keys.items():
            return_value[option] = self.na_helper.get_value_for_bool(from_zapi=True, value=sis_info.get_child_content(zapi_key))
        for option, zapi_key in self.na_helper.zapi_str_keys.items():
            return_value[option] = sis_info.get_child_content(zapi_key)
        return return_value
    return None