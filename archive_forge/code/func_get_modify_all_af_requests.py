from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_all_af_requests(self, conf_addr_fams, vrf_name):
    requests = []
    for conf_addr_fam in conf_addr_fams:
        conf_afi = conf_addr_fam.get('afi', None)
        conf_safi = conf_addr_fam.get('safi', None)
        if conf_afi and conf_safi:
            requests.extend(self.get_modify_single_af_request(vrf_name, conf_afi, conf_safi, conf_addr_fam))
    return requests