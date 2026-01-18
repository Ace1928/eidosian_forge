from __future__ import absolute_import, division, print_function
from ..module_utils.api import NIOS_DTC_POOL
from ..module_utils.api import WapiModule
from ..module_utils.api import normalize_ib_spec
from ansible.module_utils.basic import AnsibleModule
def monitors_transform(module):
    monitor_list = list()
    if module.params['monitors']:
        for monitor in module.params['monitors']:
            monitor_obj = wapi.get_object('dtc:monitor:' + monitor['type'], {'name': monitor['name']})
            if monitor_obj:
                monitor_list.append(monitor_obj[0]['_ref'])
            else:
                module.fail_json(msg='monitor %s cannot be found.' % monitor)
    return monitor_list