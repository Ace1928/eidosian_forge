from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_client_list(self, cfg):
    client_lst = []
    client_lists = cfg
    client_dict = {}
    if isinstance(client_lists, dict):
        client_dict['name'] = client_lists['name']
        if 'client-address-list' in client_lists.keys():
            client_addresses = client_lists['client-address-list']
            client_address_lst = []
            if isinstance(client_addresses, dict):
                client_address_lst.append(self.get_client_address(client_addresses))
            else:
                for address in client_addresses:
                    client_address_lst.append(self.get_client_address(address))
            if client_address_lst:
                client_dict['addresses'] = client_address_lst
        client_lst.append(client_dict)
    else:
        for client in client_lists:
            client_dict['name'] = client['name']
            if 'client-address-list' in client.keys():
                client_addresses = client['client-address-list']
                client_address_lst = []
                if isinstance(client_addresses, dict):
                    client_address_lst.append(self.get_client_address(client_addresses))
                else:
                    for address in client_addresses:
                        client_address_lst.append(self.get_client_address(address))
                if client_address_lst:
                    client_dict['addresses'] = client_address_lst
            client_lst.append(client_dict)
            client_dict = {}
    return client_lst