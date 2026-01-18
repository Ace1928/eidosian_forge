from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_get(self, cluster):
    """
        Get current cluster peer info
        :param cluster: type of cluster (source or destination)
        :return: Dictionary of current cluster peer details if query successful, else return None
        """
    if self.use_rest:
        return self.cluster_peer_get_rest(cluster)
    cluster_peer_get_iter = self.cluster_peer_get_iter(cluster)
    result, cluster_info = (None, dict())
    if cluster == 'source':
        server = self.server
    else:
        server = self.dest_server
    try:
        result = server.invoke_successfully(cluster_peer_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching cluster peer %s: %s' % (cluster, to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        cluster_peer_info = result.get_child_by_name('attributes-list').get_child_by_name('cluster-peer-info')
        cluster_info['cluster_name'] = cluster_peer_info.get_child_content('cluster-name')
        peers = cluster_peer_info.get_child_by_name('peer-addresses')
        cluster_info['peer-addresses'] = [peer.get_content() for peer in peers.get_children()]
        return cluster_info
    return None