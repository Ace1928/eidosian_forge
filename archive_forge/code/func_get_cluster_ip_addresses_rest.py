from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_ip_addresses_rest(self, cluster_ip_address):
    """ get list of IP addresses for this cluster
            return:
                a list of dictionaries
        """
    if_infos = []
    records = self.get_cluster_nodes_rest()
    for record in records:
        for interface in record.get('cluster_interfaces', []):
            ip_address = self.na_helper.safe_get(interface, ['ip', 'address'])
            if cluster_ip_address is None or ip_address == cluster_ip_address:
                if_info = {'address': ip_address, 'home_node': record['name']}
                if_infos.append(if_info)
    return if_infos