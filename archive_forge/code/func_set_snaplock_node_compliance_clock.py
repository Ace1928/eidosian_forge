from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def set_snaplock_node_compliance_clock(self):
    """Set ONTAP snaplock compliance clock for each node"""
    if self.use_rest:
        api = 'private/cli/snaplock/compliance-clock/initialize'
        query = {'node': self.parameters['node']}
        body = {}
        dummy, error = self.rest_api.patch(api, body, query)
        if error:
            self.module.fail_json(msg=error)
    else:
        node_snaplock_clock_obj = netapp_utils.zapi.NaElement('snaplock-set-node-compliance-clock')
        node_snaplock_clock_obj.add_new_child('node', self.parameters['node'])
        try:
            result = self.server.invoke_successfully(node_snaplock_clock_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error setting snaplock compliance clock for node %s : %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
        return result