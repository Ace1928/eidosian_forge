from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cluster_body(self, modify=None, nodes=None):
    body = {}
    params = modify if modify is not None else self.parameters
    for param_key, rest_key in {'cluster_contact': 'contact', 'cluster_location': 'location', 'cluster_name': 'name', 'single_node_cluster': 'single_node_cluster', 'timezone': 'timezone'}.items():
        if param_key in params:
            body[rest_key] = params[param_key]
    if nodes:
        body['nodes'] = nodes
    return body