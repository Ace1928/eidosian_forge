from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_broadcast_domain(self, ports=None):
    """
        Creates a new broadcast domain
        """
    if self.use_rest:
        return self.create_broadcast_domain_rest(ports)
    domain_obj = netapp_utils.zapi.NaElement('net-port-broadcast-domain-create')
    domain_obj.add_new_child('broadcast-domain', self.parameters['name'])
    if self.parameters.get('ipspace'):
        domain_obj.add_new_child('ipspace', self.parameters['ipspace'])
    if self.parameters.get('mtu'):
        domain_obj.add_new_child('mtu', str(self.parameters['mtu']))
    if self.parameters.get('ports'):
        ports_obj = netapp_utils.zapi.NaElement('ports')
        domain_obj.add_child_elem(ports_obj)
        for port in self.parameters['ports']:
            ports_obj.add_new_child('net-qualified-port-name', port)
    try:
        self.server.invoke_successfully(domain_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating broadcast domain %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())