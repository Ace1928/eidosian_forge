from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_storage_failover(self, current):
    """
        Modifies storage failover for a specified node
        """
    if self.use_rest:
        api = 'cluster/nodes'
        body = {'ha': {'enabled': self.parameters['is_enabled']}}
        dummy, error = rest_generic.patch_async(self.rest_api, api, current['uuid'], body)
        if error:
            self.module.fail_json(msg=error)
    else:
        if self.parameters['state'] == 'present':
            cf_service = 'cf-service-enable'
        else:
            cf_service = 'cf-service-disable'
        storage_failover_modify = netapp_utils.zapi.NaElement(cf_service)
        storage_failover_modify.add_new_child('node', self.parameters['node_name'])
        try:
            self.server.invoke_successfully(storage_failover_modify, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying storage failover for node %s: %s' % (self.parameters['node_name'], to_native(error)), exception=traceback.format_exc())