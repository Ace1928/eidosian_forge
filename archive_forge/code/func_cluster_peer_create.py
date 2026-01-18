from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_create(self, cluster):
    """
        Create a cluster peer on source or destination
        For source cluster, peer addresses = destination inter-cluster LIFs and vice-versa
        :param cluster: type of cluster (source or destination)
        :return: None
        """
    if self.use_rest:
        return self.cluster_peer_create_rest(cluster)
    cluster_peer_create = netapp_utils.zapi.NaElement.create_node_with_children('cluster-peer-create')
    if self.parameters.get('passphrase') is not None:
        cluster_peer_create.add_new_child('passphrase', self.parameters['passphrase'])
    peer_addresses = netapp_utils.zapi.NaElement('peer-addresses')
    server, peer_address = self.get_server_and_peer_address(cluster)
    for each in peer_address:
        peer_addresses.add_new_child('remote-inet-address', each)
    cluster_peer_create.add_child_elem(peer_addresses)
    if self.parameters.get('encryption_protocol_proposed') is not None:
        cluster_peer_create.add_new_child('encryption-protocol-proposed', self.parameters['encryption_protocol_proposed'])
    if self.parameters.get('ipspace') is not None:
        cluster_peer_create.add_new_child('ipspace-name', self.parameters['ipspace'])
    try:
        server.invoke_successfully(cluster_peer_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating cluster peer %s: %s' % (peer_address, to_native(error)), exception=traceback.format_exc())