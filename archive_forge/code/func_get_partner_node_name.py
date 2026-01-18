from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_partner_node_name(self):
    """
        return: partner_node_name, str
        """
    if self.use_rest:
        api = '/cluster/nodes'
        query = {'ha.partners.name': self.parameters['node']}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_more_records(api, message, error)
        if error:
            self.module.fail_json(msg=error)
        return records[0]['name'] if records else None
    else:
        partner_name = None
        cf_status = netapp_utils.zapi.NaElement('cf-status')
        cf_status.add_new_child('node', self.parameters['node'])
        try:
            result = self.server.invoke_successfully(cf_status, True)
            if result.get_child_by_name('partner-name'):
                partner_name = result.get_child_content('partner-name')
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting partner name for node %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
        return partner_name