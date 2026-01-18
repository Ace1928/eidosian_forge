from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_if_grp_rest(self, ports, allow_partial_match, force=False):
    api = 'network/ethernet/ports'
    query = {'type': 'lag', 'node.name': self.parameters['node']}
    fields = 'name,node,uuid,broadcast_domain,lag'
    error = None
    if not self.current_records or force:
        self.current_records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    if self.current_records:
        current_ifgrp = self.get_if_grp_current(self.current_records, ports)
        if current_ifgrp:
            exact_match = self.check_exact_match(ports, current_ifgrp['ports'])
            if exact_match or allow_partial_match:
                return (current_ifgrp, exact_match)
    return (None, None)