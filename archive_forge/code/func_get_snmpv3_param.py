from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_snmpv3_param(self, cfg):
    cfg_dict = {}
    cfg_dict['name'] = cfg.get('name')
    if 'notify-filter' in cfg.keys():
        cfg_dict['notify_filter'] = cfg.get('notify-filter')
    if 'parameters' in cfg.keys():
        param_dict = {}
        parameters = cfg.get('parameters')
        for key in parameters.keys():
            param_dict[key.replace('-', '_')] = parameters.get(key)
        cfg_dict['parameters'] = param_dict
    return cfg_dict