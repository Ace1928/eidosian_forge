from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def create_fpolicy_policy(self):
    """
        Create an FPolicy policy.
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy'
        body = {'vserver': self.parameters['vserver'], 'policy-name': self.parameters['name'], 'events': self.parameters['events']}
        for parameter in ('engine', 'allow_privileged_access', 'is_mandatory', 'is_passthrough_read_enabled', 'privileged_user_name'):
            if parameter in self.parameters:
                body[parameter.replace('_', '-')] = self.parameters[parameter]
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_policy_obj = netapp_utils.zapi.NaElement('fpolicy-policy-create')
        fpolicy_policy_obj.add_new_child('policy-name', self.parameters['name'])
        if 'is_mandatory' in self.parameters:
            fpolicy_policy_obj.add_new_child('is-mandatory', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_mandatory']))
        if 'engine' in self.parameters:
            fpolicy_policy_obj.add_new_child('engine-name', self.parameters['engine'])
        if 'allow_privileged_access' in self.parameters:
            fpolicy_policy_obj.add_new_child('allow-privileged-access', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['allow_privileged_access']))
        if 'is_passthrough_read_enabled' in self.parameters:
            fpolicy_policy_obj.add_new_child('is-passthrough-read-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_passthrough_read_enabled']))
        events_obj = netapp_utils.zapi.NaElement('events')
        for event in self.parameters['events']:
            events_obj.add_new_child('event-name', event)
        fpolicy_policy_obj.add_child_elem(events_obj)
        if 'privileged_user_name' in self.parameters:
            fpolicy_policy_obj.add_new_child('privileged-user-name', self.parameters['privileged_user_name'])
        try:
            self.server.invoke_successfully(fpolicy_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating fPolicy policy %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())