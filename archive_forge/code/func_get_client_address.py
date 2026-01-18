from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_client_address(self, cfg):
    cfg_dict = {}
    cfg_dict['address'] = cfg['name']
    if 'restrict' in cfg.keys():
        cfg_dict['restrict'] = True
    return cfg_dict