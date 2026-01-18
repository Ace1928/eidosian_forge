from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_delete_rest(self, cluster, uuid):
    server = self.rest_api if cluster == 'source' else self.dst_rest_api
    dummy, error = rest_generic.delete_async(server, 'cluster/peers', uuid)
    if error:
        self.module.fail_json(msg=error)