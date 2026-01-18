from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
def fortios_monitor(fos):
    valid, result = validate_parameters(fos)
    if not valid:
        return (True, False, result)
    params = fos._module.params
    selector = params['selector']
    selector_params = params['params']
    resp = fos.monitor_post(module_selectors_defs[selector]['url'], vdom=params['vdom'], data=selector_params)
    return (not is_successful_status(resp), False, resp)