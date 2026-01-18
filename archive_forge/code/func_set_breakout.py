from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def set_breakout(module, rest_obj, breakout_config, breakout_capability, interface_id, device_id):
    """
    Configuration the breakout feature for given option.
    :param module: ansible module arguments.
    :param rest_obj: rest object for making requests.
    :param breakout_config: Existing breakout configuration.
    :param breakout_capability: Available breakout configuration.
    :param interface_id: port number with service tag
    :param device_id: device id
    :return: rest object
    """
    breakout_type, response = (module.params['breakout_type'], {})
    payload = get_breakout_payload(device_id, breakout_type, interface_id)
    if breakout_config == 'HardwareDefault' and (not breakout_type == 'HardwareDefault'):
        for config in breakout_capability:
            if breakout_type == config['Type']:
                check_mode(module, changes=True)
                response = rest_obj.invoke_request('POST', JOB_URI, data=payload)
                break
        else:
            supported_type = ', '.join((i['Type'] for i in breakout_capability))
            module.fail_json(msg='Invalid breakout type: {0}, supported values are {1}.'.format(breakout_type, supported_type))
    elif not breakout_config == 'HardwareDefault' and breakout_type == 'HardwareDefault':
        check_mode(module, changes=True)
        response = rest_obj.invoke_request('POST', JOB_URI, data=payload)
    elif breakout_config == breakout_type:
        check_mode(module, changes=False)
        module.exit_json(msg='The port is already configured with the selected breakout configuration.')
    else:
        module.fail_json(msg='Device does not support changing a port breakout configuration to different breakout type. Configure the port to HardwareDefault and retry the operation.')
    return response