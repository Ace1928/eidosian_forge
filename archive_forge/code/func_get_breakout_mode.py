from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_breakout_mode(module, name):
    response = None
    mode = None
    component_name = name
    if '/' in name:
        component_name = name.replace('/', '%2f')
    url = 'data/openconfig-platform:components/component=%s' % component_name
    request = [{'path': url, 'method': GET}]
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        try:
            json_obj = json.loads(str(exc).replace("'", '"'))
            if json_obj and isinstance(json_obj, dict) and (404 == json_obj['code']):
                response = None
            else:
                module.fail_json(msg=str(exc), code=exc.code)
        except Exception as err:
            module.fail_json(msg=str(exc), code=exc.code)
    if response and 'openconfig-platform:component' in response[0][1]:
        raw_port_breakout = response[0][1]['openconfig-platform:component'][0]
        port_name = raw_port_breakout.get('name', None)
        port_data = raw_port_breakout.get('port', None)
        if port_name and port_data and ('openconfig-platform-port:breakout-mode' in port_data):
            if 'groups' in port_data['openconfig-platform-port:breakout-mode']:
                group = port_data['openconfig-platform-port:breakout-mode']['groups']['group'][0]
                if 'config' in group:
                    cfg = group.get('config', None)
                    breakout_speed = cfg.get('breakout-speed', None)
                    num_breakouts = cfg.get('num-breakouts', None)
                    if breakout_speed and num_breakouts:
                        speed = breakout_speed.replace('openconfig-if-ethernet:SPEED_', '')
                        speed = speed.replace('GB', 'G')
                        mode = str(num_breakouts) + 'x' + speed
    return mode