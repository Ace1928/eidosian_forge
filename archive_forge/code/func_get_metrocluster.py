from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_metrocluster(self):
    attrs = None
    api = 'cluster/metrocluster'
    options = {'fields': '*'}
    message, error = self.rest_api.get(api, options)
    if error:
        self.module.fail_json(msg=error)
    if message is not None:
        local = message['local']
        if local['configuration_state'] != 'not_configured':
            attrs = {'configuration_state': local['configuration_state'], 'partner_cluster_reachable': local['partner_cluster_reachable'], 'partner_cluster_name': local['cluster']['name']}
    return attrs