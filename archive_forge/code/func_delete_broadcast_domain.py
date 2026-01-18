from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_broadcast_domain(self, broadcast_domain=None, current=None):
    """
        Deletes a broadcast domain
        """
    if self.use_rest:
        if 'ports' in current:
            self.remove_broadcast_domain_ports_rest(current['ports'], current['ipspace'])
        api = 'network/ethernet/broadcast-domains'
        dummy, error = rest_generic.delete_async(self.rest_api, api, current['uuid'])
        if error:
            self.module.fail_json(msg=error)
    else:
        if broadcast_domain is None:
            broadcast_domain = self.parameters['name']
        domain_obj = netapp_utils.zapi.NaElement('net-port-broadcast-domain-destroy')
        domain_obj.add_new_child('broadcast-domain', broadcast_domain)
        if self.parameters.get('ipspace'):
            domain_obj.add_new_child('ipspace', self.parameters['ipspace'])
        try:
            self.server.invoke_successfully(domain_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting broadcast domain %s: %s' % (broadcast_domain, to_native(error)), exception=traceback.format_exc())