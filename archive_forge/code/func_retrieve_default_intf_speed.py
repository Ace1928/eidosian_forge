from __future__ import absolute_import, division, print_function
import traceback
import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def retrieve_default_intf_speed(module, intf_name):
    dft_intf_speed = 'SPEED_DEFAULT'
    method = 'get'
    sonic_port_url = 'data/sonic-port:sonic-port/PORT/PORT_LIST=%s'
    sonic_port_vs_url = (sonic_port_url + '/valid_speeds') % quote(intf_name, safe='')
    request = {'path': sonic_port_vs_url, 'method': method}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    if 'sonic-port:valid_speeds' in response[0][1]:
        v_speeds = response[0][1].get('sonic-port:valid_speeds', '')
        v_speeds_list = v_speeds.split(',')
        v_speeds_int_list = []
        for vs in v_speeds_list:
            v_speeds_int_list.append(int(vs))
        dft_speed_int = 0
        if v_speeds_int_list:
            dft_speed_int = max(v_speeds_int_list)
        dft_intf_speed = intf_speed_map.get(dft_speed_int, 'SPEED_DEFAULT')
    if dft_intf_speed == 'SPEED_DEFAULT':
        module.fail_json(msg='Unable to retireve default port speed for the interface {0}'.format(intf_name))
    return dft_intf_speed