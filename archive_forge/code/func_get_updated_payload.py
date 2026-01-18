from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_updated_payload(rest_obj, module):
    params = module.params
    resp = rest_obj.invoke_request('GET', WEBSERVER_CONFIG)
    current_setting = resp.json_data
    port_changed = 0
    cp = current_setting.copy()
    klist = cp.keys()
    for k in klist:
        if str(k).lower().startswith('@odata'):
            current_setting.pop(k)
    diff = 0
    webserver_payload_map = {'webserver_port': 'PortNumber', 'webserver_timeout': 'TimeOut'}
    for config, pload in webserver_payload_map.items():
        pval = params.get(config)
        if pval is not None:
            if current_setting.get(pload) != pval:
                current_setting[pload] = pval
                if pload == 'PortNumber':
                    port_changed = pval
                diff += 1
    if diff == 0:
        if module.check_mode:
            module.exit_json(msg='No changes found to be applied to the web server.')
        module.exit_json(msg='No changes made to the web server configuration as the entered values are the same as the current configuration.', webserver_configuration=current_setting)
    if module.check_mode:
        module.exit_json(changed=True, msg='Changes found to be applied to the web server.')
    return (current_setting, port_changed)