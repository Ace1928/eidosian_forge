from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_active_directory(self):
    if self.use_rest:
        return self.get_active_directory_rest()
    active_directory_iter = netapp_utils.zapi.NaElement('active-directory-account-get-iter')
    active_directory_info = netapp_utils.zapi.NaElement('active-directory-account-config')
    active_directory_info.add_new_child('account-name', self.parameters['account_name'])
    active_directory_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(active_directory_info)
    active_directory_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(active_directory_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error searching for Active Directory %s: %s' % (self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())
    record = {}
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        account_info = result.get_child_by_name('attributes-list').get_child_by_name('active-directory-account-config')
        for zapi_key, key in (('account-name', 'account_name'), ('domain', 'domain'), ('organizational-unit', 'organizational_unit')):
            value = account_info.get_child_content(zapi_key)
            if value is not None:
                record[key] = value
        for key, value in record.items():
            if key in self.parameters and self.parameters[key].lower() == value.lower():
                record[key] = self.parameters[key]
    return record or None