from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_aigp(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    aigp_dict = {}
    aigp = cfg.get('aigp')
    if aigp and 'disable' in aigp.keys():
        aigp_dict['disable'] = True
    else:
        aigp_dict['set'] = True
    return aigp_dict