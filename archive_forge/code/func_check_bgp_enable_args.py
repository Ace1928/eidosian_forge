from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_bgp_enable_args(**kwargs):
    """ check_bgp_enable_args """
    module = kwargs['module']
    need_cfg = False
    as_number = module.params['as_number']
    if as_number:
        if len(as_number) > 11 or len(as_number) == 0:
            module.fail_json(msg='Error: The len of as_number %s is out of [1 - 11].' % as_number)
        else:
            need_cfg = True
    return need_cfg