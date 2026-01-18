from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_broadcast_domain_ports(self):
    """
        Return details about the broadcast domain ports.
        :return: Details about the broadcast domain ports. [] if not found.
        :rtype: list
        """
    if self.use_rest:
        return self.get_broadcast_domain_ports_rest()
    domain_get_iter = netapp_utils.zapi.NaElement('net-port-broadcast-domain-get-iter')
    broadcast_domain_info = netapp_utils.zapi.NaElement('net-port-broadcast-domain-info')
    broadcast_domain_info.add_new_child('broadcast-domain', self.parameters['resource_name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(broadcast_domain_info)
    domain_get_iter.add_child_elem(query)
    result = self.server.invoke_successfully(domain_get_iter, True)
    ports = []
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        domain_info = result.get_child_by_name('attributes-list').get_child_by_name('net-port-broadcast-domain-info')
        domain_ports = domain_info.get_child_by_name('ports')
        if domain_ports is not None:
            ports = [port.get_child_content('port') for port in domain_ports.get_children()]
    return ports