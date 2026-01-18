from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_all_vnis_request(self, vrf_name, conf_afi, conf_safi, conf_vnis):
    requests = []
    for vni in conf_vnis:
        requests.append(self.get_delete_vni_request(vrf_name, conf_afi, conf_safi, vni['vni_number']))
    return requests