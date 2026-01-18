from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def get_current_payload(fabric_details, rest_obj):
    """
    extract payload from existing fabric details, which is
     obtained from GET request of existing fabric, to match with payload created
    :param fabric_details: dict - specified fabric details
    :return: dict
    """
    if fabric_details.get('OverrideLLDPConfiguration') and fabric_details.get('OverrideLLDPConfiguration') not in ['Enabled', 'Disabled']:
        fabric_details.pop('OverrideLLDPConfiguration', None)
    payload = {'Id': fabric_details['Id'], 'Name': fabric_details['Name'], 'Description': fabric_details.get('Description'), 'OverrideLLDPConfiguration': fabric_details.get('OverrideLLDPConfiguration'), 'FabricDesignMapping': fabric_details.get('FabricDesignMapping', []), 'FabricDesign': get_fabric_design(fabric_details['FabricDesign'].get('@odata.id'), rest_obj)}
    return dict([(k, v) for k, v in payload.items() if v])