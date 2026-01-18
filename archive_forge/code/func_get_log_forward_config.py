from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_log_forward_config(self):
    """
        gets log forward configuration
        :return: dict of log forward properties if exist, None if not
        """
    if self.use_rest:
        log_forward_config = None
        api = 'security/audit/destinations'
        query = {'fields': 'port,protocol,facility,address,verify_server', 'address': self.parameters['destination'], 'port': self.parameters['port']}
        message, error = self.rest_api.get(api, query)
        if error:
            self.module.fail_json(msg=error)
        if len(message.keys()) == 0:
            return None
        elif 'records' in message and len(message['records']) == 0:
            return None
        elif 'records' not in message:
            error = 'Unexpected response in get_security_key_manager from %s: %s' % (api, repr(message))
            self.module.fail_json(msg=error)
        log_forward_config = {'destination': message['records'][0]['address'], 'facility': message['records'][0]['facility'], 'port': message['records'][0]['port'], 'protocol': message['records'][0]['protocol'], 'verify_server': message['records'][0]['verify_server']}
        return log_forward_config
    else:
        log_forward_config = None
        log_forward_get = netapp_utils.zapi.NaElement('cluster-log-forward-get')
        log_forward_get.add_new_child('destination', self.parameters['destination'])
        log_forward_get.add_new_child('port', self.na_helper.get_value_for_int(False, self.parameters['port']))
        try:
            result = self.server.invoke_successfully(log_forward_get, True)
        except netapp_utils.zapi.NaApiError as error:
            if to_native(error.code) == '15661':
                return None
            else:
                self.module.fail_json(msg='Error getting log forward configuration for destination %s on port %s: %s' % (self.parameters['destination'], self.na_helper.get_value_for_int(False, self.parameters['port']), to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes'):
            log_forward_attributes = result.get_child_by_name('attributes')
            cluster_log_forward_info = log_forward_attributes.get_child_by_name('cluster-log-forward-info')
            log_forward_config = {'destination': cluster_log_forward_info.get_child_content('destination'), 'facility': cluster_log_forward_info.get_child_content('facility'), 'port': self.na_helper.get_value_for_int(True, cluster_log_forward_info.get_child_content('port')), 'protocol': cluster_log_forward_info.get_child_content('protocol'), 'verify_server': self.na_helper.get_value_for_bool(True, cluster_log_forward_info.get_child_content('verify-server'))}
        return log_forward_config