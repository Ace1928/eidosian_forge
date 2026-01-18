from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_port_information(module, rest_obj, device_id):
    """
    This function returns the existing breakout configuration details.
    :param module: ansible module arguments.
    :param rest_obj: rest object for making requests.
    :param device_id: device id
    :return: str, {}, str
    """
    response = rest_obj.invoke_request('GET', PORT_INFO_URI.format(device_id))
    breakout_config, breakout_capability, target_port = (None, None, module.params['target_port'])
    for each in response.json_data.get('InventoryInfo'):
        if not each['Configuration'] == 'NoBreakout' and each['Id'] == target_port:
            breakout_capability = each['PortBreakoutCapabilities']
            breakout_config = each['Configuration']
            interface_id = each['Id']
            break
    else:
        module.fail_json(msg='{0} does not support port breakout or invalid port number entered.'.format(target_port))
    return (breakout_config, breakout_capability, interface_id)