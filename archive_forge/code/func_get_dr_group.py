from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_dr_group(self):
    return_attrs = None
    for pair in self.parameters['dr_pairs']:
        api = 'cluster/metrocluster/dr-groups'
        options = {'fields': '*', 'dr_pairs.node.name': pair['node_name'], 'dr_pairs.partner.name': pair['partner_node_name'], 'partner_cluster.name': self.parameters['partner_cluster_name']}
        message, error = self.rest_api.get(api, options)
        if error:
            self.module.fail_json(msg=error)
        if 'records' in message and message['num_records'] == 0:
            continue
        elif 'records' not in message or message['num_records'] != 1:
            error = 'Unexpected response from %s: %s' % (api, repr(message))
            self.module.fail_json(msg=error)
        record = message['records'][0]
        return_attrs = {'partner_cluster_name': record['partner_cluster']['name'], 'dr_pairs': [], 'id': record['id']}
        for dr_pair in record['dr_pairs']:
            return_attrs['dr_pairs'].append({'node_name': dr_pair['node']['name'], 'partner_node_name': dr_pair['partner']['name']})
        break
    return return_attrs