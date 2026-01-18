from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_traffic_statistics(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    ts_dict = {}
    ts = cfg.get('traffic-statistics')
    if not ts:
        ts_dict['set'] = True
    else:
        if 'interval' in ts.keys():
            ts_dict['interval'] = ts.get('interval')
        if 'labeled-path' in ts.keys():
            ts_dict['labeled_path'] = True
        if 'file' in ts.keys():
            file = ts.get('file')
            file_dict = {}
            if 'files' in file.keys():
                file_dict['files'] = file.get('files')
            if 'no-world-readable' in file.keys():
                file_dict['no_world_readable'] = True
            if 'size' in file.keys():
                file_dict['size'] = file.get('size')
            if 'world-readable' in file.keys():
                file_dict['world_readable'] = True
    return ts_dict