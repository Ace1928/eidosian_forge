from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def design_node_dict_update(design_node_map):
    """
    make one level dictionary for design map for easy processing
    :param design_node_map: design node map content
    :return: dict
    """
    d = {}
    for item in design_node_map:
        if item['DesignNode'] == 'Switch-A' and item.get('PhysicalNode'):
            d.update({'PhysicalNode1': item['PhysicalNode']})
        if item['DesignNode'] == 'Switch-B' and item.get('PhysicalNode'):
            d.update({'PhysicalNode2': item['PhysicalNode']})
    return d