from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_fpolicy_event(self):
    """
        Get FPolicy event configuration if an event matching the parameters exists
        """
    return_value = None
    if self.use_rest:
        api = '/protocols/fpolicy/%s/events' % self.vserver_uuid
        query = {'fields': 'protocol,filters,file_operations,volume_monitoring'}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_more_records(api, message, error)
        if error:
            self.module.fail_json(msg=error)
        if records is not None:
            for record in records:
                if record['name'] == self.parameters['name']:
                    return_value = {}
                    for parameter in ('protocol', 'volume_monitoring'):
                        return_value[parameter] = []
                        if parameter in record:
                            return_value[parameter] = record[parameter]
                    return_value['file_operations'] = []
                    if 'file_operations' in record:
                        file_operation_list = [file_operation for file_operation, enabled in record['file_operations'].items() if enabled]
                        return_value['file_operations'] = file_operation_list
                    return_value['filters'] = []
                    if 'filters' in record:
                        filters_list = [filter for filter, enabled in record['filters'].items() if enabled]
                        return_value['filters'] = filters_list
        return return_value
    else:
        fpolicy_event_obj = netapp_utils.zapi.NaElement('fpolicy-policy-event-get-iter')
        fpolicy_event_config = netapp_utils.zapi.NaElement('fpolicy-event-options-config')
        fpolicy_event_config.add_new_child('event-name', self.parameters['name'])
        fpolicy_event_config.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(fpolicy_event_config)
        fpolicy_event_obj.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(fpolicy_event_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error searching for FPolicy policy event %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes-list'):
            fpolicy_event_attributes = result['attributes-list']['fpolicy-event-options-config']
            file_operations = []
            if fpolicy_event_attributes.get_child_by_name('file-operations'):
                for file_operation in fpolicy_event_attributes.get_child_by_name('file-operations').get_children():
                    file_operations.append(file_operation.get_content())
            filters = []
            if fpolicy_event_attributes.get_child_by_name('filter-string'):
                for filter in fpolicy_event_attributes.get_child_by_name('filter-string').get_children():
                    filters.append(filter.get_content())
            protocol = ''
            if fpolicy_event_attributes.get_child_by_name('protocol'):
                protocol = fpolicy_event_attributes.get_child_content('protocol')
            return_value = {'vserver': fpolicy_event_attributes.get_child_content('vserver'), 'name': fpolicy_event_attributes.get_child_content('event-name'), 'file_operations': file_operations, 'filters': filters, 'protocol': protocol, 'volume_monitoring': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_event_attributes.get_child_content('volume-operation'))}
        return return_value