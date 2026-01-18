from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import FortiOSHandler
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FAIL_SOCKET_MSG
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import check_legacy_fortiosapi
import json
def json_generic(data, fos):
    vdom = data['vdom']
    json_generic_data = data['json_generic']
    data = ''
    if json_generic_data['jsonbody']:
        data = json.loads(json_generic_data['jsonbody'])
    elif json_generic_data['dictbody']:
        data = json_generic_data['dictbody']
    return fos.jsonraw(json_generic_data['method'], json_generic_data['path'], data=data, specific_params=json_generic_data['specialparams'], vdom=vdom)