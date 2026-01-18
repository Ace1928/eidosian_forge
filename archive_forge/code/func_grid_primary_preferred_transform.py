from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_NSGROUP
from ..module_utils.api import normalize_ib_spec
def grid_primary_preferred_transform(module):
    for member in module.params['grid_primary']:
        clean_grid_member(member)
    return module.params['grid_primary']