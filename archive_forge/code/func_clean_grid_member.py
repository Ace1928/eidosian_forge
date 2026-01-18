from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_NSGROUP
from ..module_utils.api import normalize_ib_spec
def clean_grid_member(member):
    if member['preferred_primaries']:
        for ext in member['preferred_primaries']:
            clean_tsig(ext)
    if member['enable_preferred_primaries'] is False:
        del member['enable_preferred_primaries']
        del member['preferred_primaries']
    if member['lead'] is False:
        del member['lead']
    if member['grid_replicate'] is False:
        del member['grid_replicate']