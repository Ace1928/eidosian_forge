from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_fpolicy_policy(self):
    """
       Check if FPolicy policy exists, if it exists get the current state of the policy.
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy'
        query = {'vserver': self.parameters['vserver'], 'policy-name': self.parameters['name'], 'fields': 'events,engine,allow-privileged-access,is-mandatory,is-passthrough-read-enabled,privileged-user-name'}
        message, error = self.rest_api.get(api, query)
        if error:
            self.module.fail_json(msg=error)
        if len(message.keys()) == 0:
            return None
        if 'records' in message and len(message['records']) == 0:
            return None
        if 'records' not in message:
            error = 'Unexpected response in get_fpolicy_policy from %s: %s' % (api, repr(message))
            self.module.fail_json(msg=error)
        return_value = {'vserver': message['records'][0]['vserver'], 'name': message['records'][0]['policy_name'], 'events': message['records'][0]['events'], 'allow_privileged_access': message['records'][0]['allow_privileged_access'], 'engine': message['records'][0]['engine'], 'is_mandatory': message['records'][0]['is_mandatory'], 'is_passthrough_read_enabled': message['records'][0]['is_passthrough_read_enabled']}
        if 'privileged_user_name' in message['records'][0]:
            return_value['privileged_user_name'] = message['records'][0]['privileged_user_name']
        return return_value
    else:
        return_value = None
        fpolicy_policy_obj = netapp_utils.zapi.NaElement('fpolicy-policy-get-iter')
        fpolicy_policy_config = netapp_utils.zapi.NaElement('fpolicy-policy-info')
        fpolicy_policy_config.add_new_child('policy-name', self.parameters['name'])
        fpolicy_policy_config.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(fpolicy_policy_config)
        fpolicy_policy_obj.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(fpolicy_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error searching for fPolicy policy %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes-list'):
            fpolicy_policy_attributes = result['attributes-list']['fpolicy-policy-info']
            events = []
            if fpolicy_policy_attributes.get_child_by_name('events'):
                for event in fpolicy_policy_attributes.get_child_by_name('events').get_children():
                    events.append(event.get_content())
            return_value = {'vserver': fpolicy_policy_attributes.get_child_content('vserver'), 'name': fpolicy_policy_attributes.get_child_content('policy-name'), 'events': events, 'allow_privileged_access': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_policy_attributes.get_child_content('allow-privileged-access')), 'engine': fpolicy_policy_attributes.get_child_content('engine-name'), 'is_mandatory': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_policy_attributes.get_child_content('is-mandatory')), 'is_passthrough_read_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_policy_attributes.get_child_content('is-passthrough-read-enabled')), 'privileged_user_name': fpolicy_policy_attributes.get_child_content('privileged-user-name')}
        return return_value