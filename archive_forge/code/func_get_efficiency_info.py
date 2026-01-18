from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_efficiency_info(self, return_value):
    """
        get the name of the efficiency policy assigned to volume, as well as compression values
        if attribute does not exist, set its value to None
        :return: update return_value dict.
        """
    sis_info = netapp_utils.zapi.NaElement('sis-get-iter')
    sis_status_info = netapp_utils.zapi.NaElement('sis-status-info')
    sis_status_info.add_new_child('path', '/vol/' + self.parameters['name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(sis_status_info)
    sis_info.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(sis_info, True)
    except netapp_utils.zapi.NaApiError as error:
        if error.message.startswith('Insufficient privileges: user ') and error.message.endswith(' does not have read access to this resource'):
            self.issues.append('cannot read volume efficiency options (as expected when running as vserver): %s' % to_native(error))
            return
        self.wrap_fail_json(msg='Error fetching efficiency policy for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    for key in self.sis_keys2zapi_get:
        return_value[key] = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        sis_attributes = result.get_child_by_name('attributes-list').get_child_by_name('sis-status-info')
        for key, attr in self.sis_keys2zapi_get.items():
            value = sis_attributes.get_child_content(attr)
            if self.argument_spec[key]['type'] == 'bool':
                value = self.na_helper.get_value_for_bool(True, value)
            return_value[key] = value