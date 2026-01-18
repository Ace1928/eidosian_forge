from __future__ import absolute_import, division, print_function
from ..module_utils.api import NIOS_DTC_LBDN
from ..module_utils.api import WapiModule
from ..module_utils.api import normalize_ib_spec
from ansible.module_utils.basic import AnsibleModule
def auth_zones_transform(module):
    zone_list = list()
    if module.params['auth_zones']:
        for zone in module.params['auth_zones']:
            zone_obj = wapi.get_object('zone_auth', {'fqdn': zone})
            if zone_obj:
                zone_list.append(zone_obj[0]['_ref'])
            else:
                module.fail_json(msg='auth_zone %s cannot be found.' % zone)
    return zone_list