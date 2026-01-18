from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_get_iter(self, cluster):
    """
        Compose NaElement object to query current source cluster using peer-cluster-name and peer-addresses parameters
        :param cluster: type of cluster (source or destination)
        :return: NaElement object for cluster-get-iter with query
        """
    cluster_peer_get = netapp_utils.zapi.NaElement('cluster-peer-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    cluster_peer_info = netapp_utils.zapi.NaElement('cluster-peer-info')
    peer_lifs, peer_cluster = self.get_peer_lifs_cluster_keys(cluster)
    if self.parameters.get(peer_lifs):
        peer_addresses = netapp_utils.zapi.NaElement('peer-addresses')
        for peer in self.parameters.get(peer_lifs):
            peer_addresses.add_new_child('remote-inet-address', peer)
        cluster_peer_info.add_child_elem(peer_addresses)
    if self.parameters.get(peer_cluster):
        cluster_peer_info.add_new_child('cluster-name', self.parameters[peer_cluster])
    query.add_child_elem(cluster_peer_info)
    cluster_peer_get.add_child_elem(query)
    return cluster_peer_get