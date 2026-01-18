from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_ip_addresses(self, cluster_ip_address, ignore_error=True):
    """ get list of IP addresses for this cluster
            return:
                a list of dictionaries
        """
    if_infos = []
    zapi = netapp_utils.zapi.NaElement('net-interface-get-iter')
    if cluster_ip_address is not None:
        query = netapp_utils.zapi.NaElement('query')
        net_info = netapp_utils.zapi.NaElement('net-interface-info')
        net_info.add_new_child('address', cluster_ip_address)
        query.add_child_elem(net_info)
        zapi.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(zapi, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if ignore_error:
            return if_infos
        self.module.fail_json(msg='Error getting IP addresses: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('attributes-list'):
        for net_info in result.get_child_by_name('attributes-list').get_children():
            if net_info:
                if_info = {'address': net_info.get_child_content('address')}
                if_info['home_node'] = net_info.get_child_content('home-node')
            if_infos.append(if_info)
    return if_infos