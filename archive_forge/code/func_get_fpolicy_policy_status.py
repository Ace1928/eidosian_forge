from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_fpolicy_policy_status(self):
    """
        Check to see the status of the fPolicy policy
        :return: dict of status properties
        """
    return_value = None
    if self.use_rest:
        api = '/protocols/fpolicy'
        query = {'svm.name': self.parameters['vserver'], 'fields': 'policies'}
        message, error = self.rest_api.get(api, query)
        if error:
            self.module.fail_json(msg=error)
        records, error = rrh.check_for_0_or_more_records(api, message, error)
        if records is not None:
            for policy in records[0]['policies']:
                if policy['name'] == self.parameters['policy_name']:
                    return_value = {}
                    return_value['vserver'] = records[0]['svm']['name']
                    return_value['policy_name'] = policy['name']
                    return_value['status'] = policy['enabled']
                    break
        if not return_value:
            self.module.fail_json(msg='Error getting fPolicy policy %s for vserver %s as policy does not exist' % (self.parameters['policy_name'], self.parameters['vserver']))
        return return_value
    else:
        fpolicy_status_obj = netapp_utils.zapi.NaElement('fpolicy-policy-status-get-iter')
        fpolicy_status_info = netapp_utils.zapi.NaElement('fpolicy-policy-status-info')
        fpolicy_status_info.add_new_child('policy-name', self.parameters['policy_name'])
        fpolicy_status_info.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(fpolicy_status_info)
        fpolicy_status_obj.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(fpolicy_status_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting status for fPolicy policy %s for vserver %s: %s' % (self.parameters['policy_name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes-list'):
            fpolicy_status_attributes = result['attributes-list']['fpolicy-policy-status-info']
            return_value = {'vserver': fpolicy_status_attributes.get_child_content('vserver'), 'policy_name': fpolicy_status_attributes.get_child_content('policy-name'), 'status': self.na_helper.get_value_for_bool(True, fpolicy_status_attributes.get_child_content('status'))}
        return return_value