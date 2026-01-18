from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def create_modify_payload(module_params, fabric_id, msm_version):
    """
    payload creation for fabric management in case of create/modify operations
    :param module_params: ansible module parameters
    :param fabric_id: fabric id in case of modify operation
    :param msm_version: msm version details
    :return: dict
    """
    backup_params = dict([(k, v) for k, v in module_params.items() if v])
    _payload = {'Name': backup_params['name'], 'Description': backup_params.get('description'), 'OverrideLLDPConfiguration': backup_params.get('override_LLDP_configuration'), 'FabricDesignMapping': [], 'FabricDesign': {}}
    if backup_params.get('primary_switch_service_tag'):
        _payload['FabricDesignMapping'].append({'DesignNode': 'Switch-A', 'PhysicalNode': backup_params['primary_switch_service_tag']})
    if backup_params.get('secondary_switch_service_tag'):
        _payload['FabricDesignMapping'].append({'DesignNode': 'Switch-B', 'PhysicalNode': backup_params['secondary_switch_service_tag']})
    if backup_params.get('fabric_design'):
        _payload.update({'FabricDesign': {'Name': backup_params['fabric_design']}})
    if msm_version.startswith('1.0'):
        _payload.pop('OverrideLLDPConfiguration', None)
    if fabric_id:
        _payload['Name'] = backup_params.get('new_name', backup_params['name'])
        _payload['Id'] = fabric_id
    payload = dict([(k, v) for k, v in _payload.items() if v])
    return payload