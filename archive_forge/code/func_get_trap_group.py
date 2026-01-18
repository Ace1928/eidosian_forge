from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_trap_group(self, cfg):
    cfg_dict = {}
    cfg_dict['name'] = cfg.get('name')
    if 'categories' in cfg.keys():
        categories_dict = {}
        categories = cfg.get('categories')
        for item in categories.keys():
            if item == 'otn-alarms':
                otn_dict = {}
                otn_alarms = categories.get('otn-alarms')
                for key in otn_alarms.keys():
                    otn_dict[key.replace('-', '_')] = True
                categories_dict['otn_alarms'] = otn_dict
            else:
                categories_dict[item.replace('-', '_')] = True
        cfg_dict['categories'] = categories_dict
    if 'destination-port' in cfg.keys():
        cfg_dict['destination_port'] = cfg.get('destination-port')
    if 'routing-instance' in cfg.keys():
        cfg_dict['routing_instance'] = cfg.get('routing-instance')
    if 'version' in cfg.keys():
        cfg_dict['version'] = cfg.get('version')
    if 'targets' in cfg.keys():
        targets_lst = []
        targets = cfg.get('targets')
        if isinstance(targets, dict):
            targets_lst.append(targets['name'])
        else:
            for item in targets:
                targets_lst.append(item['name'])
        cfg_dict['targets'] = targets_lst
    return cfg_dict