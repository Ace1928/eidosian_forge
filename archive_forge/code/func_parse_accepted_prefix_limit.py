from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_accepted_prefix_limit(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    apl_dict = {}
    if 'accepted-prefix-limit' in cfg.keys():
        apl = cfg.get('accepted-prefix-limit')
    else:
        apl = cfg.get('prefix-limit')
    if 'maximum' in apl.keys():
        apl_dict['maximum'] = apl.get('maximum')
    if 'teardown' in apl.keys():
        if not apl.get('teardown'):
            apl_dict['teardown'] = True
        else:
            td = apl.get('teardown')
            if 'idle-timeout' in td.keys():
                if not td.get('idle-timeout'):
                    apl_dict['idle_timeout'] = True
                elif 'forever' in td['idle-timeout'].keys():
                    apl_dict['forever'] = True
                elif 'timeout' in td['idle-timeout'].keys():
                    apl_dict['idle_timeout_value'] = td['idle-timeout'].get('timeout')
            if 'limit-threshold' in td.keys():
                apl_dict['limit_threshold'] = td.get('limit-threshold')
    return apl_dict