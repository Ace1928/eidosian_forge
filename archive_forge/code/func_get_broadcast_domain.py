from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_broadcast_domain(self, broadcast_domain=None, ipspace=None):
    """
        Return details about the broadcast domain
        :param broadcast_domain: specific broadcast domain to get.
        :return: Details about the broadcast domain. None if not found.
        :rtype: dict
        """
    if broadcast_domain is None:
        broadcast_domain = self.parameters['name']
    if ipspace is None:
        ipspace = self.parameters.get('ipspace')
    if self.use_rest:
        return self.get_broadcast_domain_rest(broadcast_domain, ipspace)
    domain_get_iter = netapp_utils.zapi.NaElement('net-port-broadcast-domain-get-iter')
    broadcast_domain_info = netapp_utils.zapi.NaElement('net-port-broadcast-domain-info')
    broadcast_domain_info.add_new_child('broadcast-domain', broadcast_domain)
    if ipspace:
        broadcast_domain_info.add_new_child('ipspace', ipspace)
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(broadcast_domain_info)
    domain_get_iter.add_child_elem(query)
    result = self.server.invoke_successfully(domain_get_iter, True)
    domain_exists = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        domain_info = result.get_child_by_name('attributes-list').get_child_by_name('net-port-broadcast-domain-info')
        domain_name = domain_info.get_child_content('broadcast-domain')
        domain_mtu = domain_info.get_child_content('mtu')
        domain_ipspace = domain_info.get_child_content('ipspace')
        domain_ports = domain_info.get_child_by_name('ports')
        if domain_ports is not None:
            ports = [port.get_child_content('port') for port in domain_ports.get_children()]
        else:
            ports = []
        domain_exists = {'domain-name': domain_name, 'mtu': int(domain_mtu), 'ipspace': domain_ipspace, 'ports': ports}
    return domain_exists