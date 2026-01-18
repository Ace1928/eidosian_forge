from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def create_metrocluster(self):
    api = 'cluster/metrocluster'
    options = {}
    dr_pairs = []
    for pair in self.parameters['dr_pairs']:
        dr_pairs.append({'node': {'name': pair['node_name']}, 'partner': {'name': pair['partner_node_name']}})
    partner_cluster = {'name': self.parameters['partner_cluster_name']}
    data = {'dr_pairs': dr_pairs, 'partner_cluster': partner_cluster}
    message, error = self.rest_api.post(api, data, options)
    if error is not None:
        self.module.fail_json(msg='%s' % error)
    message, error = self.rest_api.wait_on_job(message['job'])
    if error:
        self.module.fail_json(msg='%s' % error)