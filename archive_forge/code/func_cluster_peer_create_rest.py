from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_create_rest(self, cluster):
    api = 'cluster/peers'
    body = {}
    if self.parameters.get('passphrase') is not None:
        body['authentication.passphrase'] = self.parameters['passphrase']
    elif cluster == 'source':
        body['authentication.generate_passphrase'] = True
    elif cluster == 'destination':
        body['authentication.passphrase'] = self.generated_passphrase
    server, peer_address = self.get_server_and_peer_address(cluster)
    body['remote.ip_addresses'] = peer_address
    if self.parameters.get('encryption_protocol_proposed') is not None:
        body['encryption.proposed'] = self.parameters['encryption_protocol_proposed']
    else:
        body['encryption.proposed'] = 'none'
    if self.parameters.get('ipspace') is not None:
        body['ipspace.name'] = self.parameters['ipspace']
    response, error = rest_generic.post_async(server, api, body)
    if error:
        self.module.fail_json(msg=error)
    if response and cluster == 'source' and ('passphrase' not in self.parameters):
        for record in response['records']:
            self.generated_passphrase = record['authentication']['passphrase']