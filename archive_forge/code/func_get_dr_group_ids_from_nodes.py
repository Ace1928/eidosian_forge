from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_dr_group_ids_from_nodes(self):
    delete_ids = []
    for pair in self.parameters['dr_pairs']:
        api = 'cluster/metrocluster/nodes'
        options = {'fields': '*', 'node.name': pair['node_name']}
        message, error = self.rest_api.get(api, options)
        if error:
            self.module.fail_json(msg=error)
        if 'records' in message and message['num_records'] == 0:
            continue
        elif 'records' not in message or message['num_records'] != 1:
            error = 'Unexpected response from %s: %s' % (api, repr(message))
            self.module.fail_json(msg=error)
        record = message['records'][0]
        if int(record['dr_group_id']) not in delete_ids:
            delete_ids.append(int(record['dr_group_id']))
    return delete_ids