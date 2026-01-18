from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def modify_fpolicy_event(self, modify):
    """
        Modify an FPolicy policy event
        :return: nothing
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy/event'
        query = {'vserver': self.parameters['vserver'], 'event-name': self.parameters['name']}
        body = {}
        for parameter in ('protocol', 'filters', 'file_operations'):
            if parameter in modify:
                body[parameter] = modify[parameter]
            elif parameter in self.parameters:
                body[parameter] = self.parameters[parameter]
        if 'volume_monitoring' in modify:
            body['volume-operation'] = modify['volume_monitoring']
        dummy, error = self.rest_api.patch(api, body, query)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_event_obj = netapp_utils.zapi.NaElement('fpolicy-policy-event-modify')
        fpolicy_event_obj.add_new_child('event-name', self.parameters['name'])
        if 'file_operations' in self.parameters:
            file_operation_obj = netapp_utils.zapi.NaElement('file-operations')
            for file_operation in self.parameters['file_operations']:
                file_operation_obj.add_new_child('fpolicy-operation', file_operation)
            fpolicy_event_obj.add_child_elem(file_operation_obj)
        if 'filters' in self.parameters:
            filter_string_obj = netapp_utils.zapi.NaElement('filter-string')
            for filter in self.parameters['filters']:
                filter_string_obj.add_new_child('fpolicy-filter', filter)
            fpolicy_event_obj.add_child_elem(filter_string_obj)
        if 'protocol' in self.parameters:
            fpolicy_event_obj.add_new_child('protocol', self.parameters['protocol'])
        if 'volume_monitoring' in self.parameters:
            fpolicy_event_obj.add_new_child('volume-operation', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['volume_monitoring']))
        try:
            self.server.invoke_successfully(fpolicy_event_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying fPolicy policy event %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())