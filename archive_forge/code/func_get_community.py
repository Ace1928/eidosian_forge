from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_community(self, cfg):
    cfg_dict = {}
    cfg_dict['name'] = cfg['name']
    if 'authorization' in cfg.keys():
        cfg_dict['authorization'] = cfg.get('authorization')
    if 'client-list-name' in cfg.keys():
        cfg_dict['client_list_name'] = cfg.get('client-list-name')
    if 'clients' in cfg.keys():
        client_lst = []
        if isinstance(cfg.get('clients'), dict):
            client_lst.append(self.get_client_address(cfg.get('clients')))
        else:
            clients = cfg.get('clients')
            for item in clients:
                client_lst.append(self.get_client_address(item))
        cfg_dict['clients'] = client_lst
    if 'routing-instance' in cfg.keys():
        rinst_lst = []
        rinst_lists = cfg.get('routing-instance')
        if isinstance(rinst_lists, dict):
            rinst_lst.append(self.get_routing_instance(rinst_lists))
        else:
            for item in rinst_lists:
                rinst_lst.append(self.get_routing_instance(item))
        if rinst_lst:
            cfg_dict['routing_instances'] = rinst_lst
    if 'view' in cfg.keys():
        cfg_dict['view'] = cfg.get('view')
    return cfg_dict