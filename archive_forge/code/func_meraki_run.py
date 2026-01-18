from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def meraki_run(meraki):
    """Perform API calls and generate responses based on the 'state' param."""
    meraki_run_func = _meraki_run_func_lookup(meraki.params['state'])
    meraki_run_func(meraki)