from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_snmpv3_comm(self, cfg):
    cfg_dict = {}
    cfg_dict['community_index'] = cfg['name']
    if 'context' in cfg.keys():
        cfg_dict['context'] = cfg.get('context')
    if 'tag' in cfg.keys():
        cfg_dict['tag'] = cfg.get('tag')
    if 'security-name' in cfg.keys():
        cfg_dict['security_name'] = cfg.get('security-name')
    if 'community-name' in cfg.keys():
        cfg_dict['community_name'] = cfg.get('community-name')
    return cfg_dict