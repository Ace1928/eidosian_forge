from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_entropy_label(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    el_dict = {}
    el = cfg.get('entropy-label')
    if not el:
        el_dict['set'] = True
    else:
        if 'import' in el.keys():
            el_dict['import'] = el.get('import')
        if 'no-next-hop-validation' in el.keys():
            el_dict['no_next_hop_validation'] = True
    return el_dict