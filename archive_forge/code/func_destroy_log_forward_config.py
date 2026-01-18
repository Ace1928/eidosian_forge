from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def destroy_log_forward_config(self):
    """
        Delete a log forward configuration
        :return: nothing
        """
    if self.use_rest:
        api = 'security/audit/destinations/%s/%s' % (self.parameters['destination'], self.parameters['port'])
        body = None
        query = {'return_timeout': 3}
        dummy, error = self.rest_api.delete(api, body, query)
        if error:
            self.module.fail_json(msg=error)
    else:
        log_forward_config_obj = netapp_utils.zapi.NaElement('cluster-log-forward-destroy')
        log_forward_config_obj.add_new_child('destination', self.parameters['destination'])
        log_forward_config_obj.add_new_child('port', self.na_helper.get_value_for_int(False, self.parameters['port']))
        try:
            self.server.invoke_successfully(log_forward_config_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error destroying log forward destination %s on port %s: %s' % (self.parameters['destination'], self.na_helper.get_value_for_int(False, self.parameters['port']), to_native(error)), exception=traceback.format_exc())