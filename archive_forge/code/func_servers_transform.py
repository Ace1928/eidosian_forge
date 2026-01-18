from __future__ import absolute_import, division, print_function
from ..module_utils.api import NIOS_DTC_POOL
from ..module_utils.api import WapiModule
from ..module_utils.api import normalize_ib_spec
from ansible.module_utils.basic import AnsibleModule
def servers_transform(module):
    server_list = list()
    if module.params['servers']:
        for server in module.params['servers']:
            server_obj = wapi.get_object('dtc:server', {'name': server['server']})
            if server_obj:
                server_list.append({'server': server_obj[0]['_ref'], 'ratio': server['ratio']})
            else:
                module.fail_json(msg='Server %s cannot be found.' % server)
    return server_list