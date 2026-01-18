from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def is_option_set(self):
    """
        Checks to see if an option is set or not
        :return: If option is set return True, else return False
        """
    option_obj = netapp_utils.zapi.NaElement('options-get-iter')
    options_info = netapp_utils.zapi.NaElement('option-info')
    if self.parameters.get('name') is not None:
        options_info.add_new_child('name', self.parameters['name'])
    if self.parameters.get('value') is not None:
        options_info.add_new_child('value', self.parameters['value'])
    if 'vserver' in self.parameters.keys():
        if self.parameters['vserver'] is not None:
            options_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(options_info)
    option_obj.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(option_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error finding option: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        return True
    return False