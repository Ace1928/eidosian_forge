from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_add_path(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    ap_dict = {}
    ap = cfg.get('add-path')
    if 'receive' in ap.keys():
        ap_dict['receive'] = True
    if 'send' in ap.keys():
        send = ap.get('send')
        s_dict = {}
        if 'include-backup-path' in send.keys():
            s_dict['include_backup_path'] = send.get('include-backup-path')
        if 'path-count' in send.keys():
            s_dict['path_count'] = send.get('path-count')
        if 'multipath' in send.keys():
            s_dict['multipath'] = True
        if 'path-selection-mode' in send.keys():
            psm = send.get('path-selection-mode')
            psm_dict = {}
            if 'all-paths' in psm.keys():
                psm_dict['all_paths'] = True
            if 'equal-cost-paths' in psm.keys():
                psm_dict['equal_cost_paths'] = True
            s_dict['path_selection_mode'] = psm_dict
        if 'prefix-policy' in send.keys():
            s_dict['prefix_policy'] = send.get('prefix-policy')
        ap_dict['send'] = s_dict
    return ap_dict