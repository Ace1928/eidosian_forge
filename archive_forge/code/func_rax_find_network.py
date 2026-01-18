from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_find_network(module, rax_module, network):
    """Find a cloud network by ID or name"""
    cnw = rax_module.cloud_networks
    try:
        UUID(network)
    except ValueError:
        if network.lower() == 'public':
            return cnw.get_server_networks(PUBLIC_NET_ID)
        elif network.lower() == 'private':
            return cnw.get_server_networks(SERVICE_NET_ID)
        else:
            try:
                network_obj = cnw.find_network_by_label(network)
            except (rax_module.exceptions.NetworkNotFound, rax_module.exceptions.NetworkLabelNotUnique):
                module.fail_json(msg='No matching network found (%s)' % network)
            else:
                return cnw.get_server_networks(network_obj)
    else:
        return cnw.get_server_networks(network)