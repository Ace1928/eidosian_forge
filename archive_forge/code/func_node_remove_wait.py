from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def node_remove_wait(self):
    """ wait for node name or clister IP address to disappear """
    if self.use_rest:
        return
    node_name = self.parameters.get('node_name')
    node_ip = self.parameters.get('cluster_ip_address')
    retries = self.parameters['time_out']
    while retries > 0:
        retries = retries - 10
        if node_name is not None and node_name not in self.get_cluster_nodes():
            return
        if node_ip is not None and self.get_cluster_ip_address(node_ip) is None:
            return
        time.sleep(10)
    self.module.fail_json(msg='Timeout waiting for node to be removed from cluster.')