from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def merge_payload(modify_payload, current_payload, module):
    """
    :param modify_payload: payload created to update existing setting
    :param current_payload: already existing payload for specified fabric
    :param module: Ansible module object
    :return: bool - compare existing and requested setting values of fabric in case of modify operations
    if both are same return True
    """
    _current_payload = dict(current_payload)
    _current_payload.update(modify_payload)
    if modify_payload.get('FabricDesign') and current_payload.get('FabricDesign'):
        _current_payload['FabricDesign'].update(modify_payload['FabricDesign'])
    elif modify_payload.get('FabricDesign') and (not current_payload.get('FabricDesign')):
        _current_payload['FabricDesign'] = modify_payload['FabricDesign']
    fabric_design_map_list = fabric_design_map_payload_creation(modify_payload.get('FabricDesignMapping', []), current_payload.get('FabricDesignMapping', []), module)
    if fabric_design_map_list:
        _current_payload.update({'FabricDesignMapping': fabric_design_map_list})
    return _current_payload