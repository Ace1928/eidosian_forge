from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_desired(module):
    return {'flush_routes': module.params['flush_routes'], 'enforce_rtr_alert': module.params['enforce_rtr_alert']}