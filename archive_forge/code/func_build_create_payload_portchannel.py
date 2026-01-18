from __future__ import absolute_import, division, print_function
import json
from copy import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def build_create_payload_portchannel(self, name, mode):
    payload_template = '{\n"openconfig-interfaces:interfaces": {"interface": [{\n"name": "{{name}}",\n"config": {\n"name": "{{name}}"\n}'
    input_data = {'name': name}
    if mode == 'static':
        payload_template += ',\n "openconfig-if-aggregation:aggregation": {\n"config": {\n"lag-type": "{{mode}}"\n}\n}\n'
        input_data['mode'] = mode.upper()
    payload_template += '}\n]\n}\n}'
    env = jinja2.Environment(autoescape=False)
    t = env.from_string(payload_template)
    intended_payload = t.render(input_data)
    ret_payload = json.loads(intended_payload)
    return ret_payload