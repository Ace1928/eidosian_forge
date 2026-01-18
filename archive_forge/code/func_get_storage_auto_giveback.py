from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_storage_auto_giveback(self):
    """
        get the storage failover giveback options for a given node
        :return: dict for options
        """
    return_value = None
    if self.use_rest:
        api = 'private/cli/storage/failover'
        query = {'fields': 'node,auto_giveback,auto_giveback_after_panic', 'node': self.parameters['name']}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_1_records(api, message, error)
        if error is None and records is not None:
            return_value = {'name': message['records'][0]['node'], 'auto_giveback_enabled': message['records'][0].get('auto_giveback'), 'auto_giveback_after_panic_enabled': message['records'][0].get('auto_giveback_after_panic')}
        if error:
            self.module.fail_json(msg=error)
        if not records:
            error = 'REST API did not return failover options for node %s' % self.parameters['name']
            self.module.fail_json(msg=error)
    else:
        storage_auto_giveback_get_iter = netapp_utils.zapi.NaElement('cf-get-iter')
        try:
            result = self.server.invoke_successfully(storage_auto_giveback_get_iter, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting auto giveback info for node %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes-list'):
            attributes_list = result.get_child_by_name('attributes-list')
            for storage_failover_info_attributes in attributes_list.get_children():
                sfo_node_info = storage_failover_info_attributes.get_child_by_name('sfo-node-info')
                node_related_info = sfo_node_info.get_child_by_name('node-related-info')
                if node_related_info.get_child_content('node') == self.parameters['name']:
                    sfo_options_info = storage_failover_info_attributes.get_child_by_name('sfo-options-info')
                    options_related_info = sfo_options_info.get_child_by_name('options-related-info')
                    sfo_giveback_options_info = options_related_info.get_child_by_name('sfo-giveback-options-info')
                    giveback_options = sfo_giveback_options_info.get_child_by_name('giveback-options')
                    return_value = {'name': node_related_info.get_child_content('node'), 'auto_giveback_enabled': self.na_helper.get_value_for_bool(True, options_related_info.get_child_content('auto-giveback-enabled')), 'auto_giveback_after_panic_enabled': self.na_helper.get_value_for_bool(True, giveback_options.get_child_content('auto-giveback-after-panic-enabled'))}
                    break
    return return_value