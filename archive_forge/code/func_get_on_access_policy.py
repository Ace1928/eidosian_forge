from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_on_access_policy(self):
    """
        Return a Vscan on Access Policy
        :return: None if there is no access policy, return the policy if there is
        """
    if self.use_rest:
        return self.get_on_access_policy_rest()
    access_policy_obj = netapp_utils.zapi.NaElement('vscan-on-access-policy-get-iter')
    access_policy_info = netapp_utils.zapi.NaElement('vscan-on-access-policy-info')
    access_policy_info.add_new_child('policy-name', self.parameters['policy_name'])
    access_policy_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(access_policy_info)
    access_policy_obj.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(access_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error searching Vscan on Access Policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())
    return_value = {}
    if result.get_child_by_name('num-records'):
        if int(result.get_child_content('num-records')) == 1:
            attributes_list = result.get_child_by_name('attributes-list')
            vscan_info = attributes_list.get_child_by_name('vscan-on-access-policy-info')
            for option, zapi_key in self.na_helper.zapi_int_keys.items():
                return_value[option] = self.na_helper.get_value_for_int(from_zapi=True, value=vscan_info.get_child_content(zapi_key))
            for option, zapi_key in self.na_helper.zapi_bool_keys.items():
                return_value[option] = self.na_helper.get_value_for_bool(from_zapi=True, value=vscan_info.get_child_content(zapi_key))
            for option, zapi_key in self.na_helper.zapi_list_keys.items():
                return_value[option] = self.na_helper.get_value_for_list(from_zapi=True, zapi_parent=vscan_info.get_child_by_name(zapi_key))
            for option, zapi_key in self.na_helper.zapi_str_keys.items():
                return_value[option] = vscan_info.get_child_content(zapi_key)
            return return_value
        elif int(result.get_child_content('num-records')) > 1:
            self.module.fail_json(msg='Mutiple Vscan on Access Policy matching %s:' % self.parameters['policy_name'])
    return None