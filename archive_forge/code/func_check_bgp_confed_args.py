from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_bgp_confed_args(**kwargs):
    """ check_bgp_confed_args """
    module = kwargs['module']
    need_cfg = False
    confed_peer_as_num = module.params['confed_peer_as_num']
    if confed_peer_as_num:
        if len(confed_peer_as_num) > 11 or len(confed_peer_as_num) == 0:
            module.fail_json(msg='Error: The len of confed_peer_as_num %s is out of [1 - 11].' % confed_peer_as_num)
        else:
            need_cfg = True
    return need_cfg