from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def modify_storage_auto_giveback(self):
    """
        Modifies storage failover giveback options for a specified node
        """
    if self.use_rest:
        api = 'private/cli/storage/failover'
        body = dict()
        query = {'node': self.parameters['name']}
        body['auto_giveback'] = self.parameters['auto_giveback_enabled']
        if 'auto_giveback_after_panic_enabled' in self.parameters:
            body['auto_giveback_after_panic'] = self.parameters['auto_giveback_after_panic_enabled']
        dummy, error = self.rest_api.patch(api, body, query)
        if error:
            self.module.fail_json(msg=error)
    else:
        storage_auto_giveback_enable = netapp_utils.zapi.NaElement('cf-modify-iter')
        attributes_info = netapp_utils.zapi.NaElement('options-related-info-modify')
        query_info = netapp_utils.zapi.NaElement('options-related-info-modify')
        attributes_info.add_new_child('node', self.parameters['name'])
        attributes_info.add_new_child('auto-giveback-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['auto_giveback_enabled']))
        if 'auto_giveback_after_panic_enabled' in self.parameters:
            sfo_give_back_options_info_modify = netapp_utils.zapi.NaElement('sfo-giveback-options-info-modify')
            give_back_options_modify = netapp_utils.zapi.NaElement('giveback-options-modify')
            give_back_options_modify.add_new_child('auto-giveback-after-panic-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['auto_giveback_after_panic_enabled']))
            sfo_give_back_options_info_modify.add_child_elem(give_back_options_modify)
            attributes_info.add_child_elem(sfo_give_back_options_info_modify)
        query = netapp_utils.zapi.NaElement('query')
        attributes = netapp_utils.zapi.NaElement('attributes')
        query.add_child_elem(query_info)
        attributes.add_child_elem(attributes_info)
        storage_auto_giveback_enable.add_child_elem(query)
        storage_auto_giveback_enable.add_child_elem(attributes)
        try:
            self.server.invoke_successfully(storage_auto_giveback_enable, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying auto giveback for node %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())