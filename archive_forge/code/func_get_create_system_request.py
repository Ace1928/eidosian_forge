from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_create_system_request(self, want, commands):
    requests = []
    host_path = 'data/openconfig-system:system/config'
    method = PATCH
    hostname_payload = self.build_create_hostname_payload(commands)
    if hostname_payload:
        request = {'path': host_path, 'method': method, 'data': hostname_payload}
        requests.append(request)
    name_path = 'data/sonic-device-metadata:sonic-device-metadata/DEVICE_METADATA/DEVICE_METADATA_LIST=localhost/intf_naming_mode'
    name_payload = self.build_create_name_payload(commands)
    if name_payload:
        request = {'path': name_path, 'method': method, 'data': name_payload}
        requests.append(request)
    anycast_path = 'data/sonic-sag:sonic-sag/SAG_GLOBAL/SAG_GLOBAL_LIST/'
    anycast_payload = self.build_create_anycast_payload(commands)
    if anycast_payload:
        request = {'path': anycast_path, 'method': method, 'data': anycast_payload}
        requests.append(request)
    return requests