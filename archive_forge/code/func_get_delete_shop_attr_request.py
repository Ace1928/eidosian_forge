from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_delete_shop_attr_request(self, remote_address, interface, vrf, local_address, attr):
    url = '%s/openconfig-bfd-ext:bfd-shop-sessions/single-hop=%s,%s,%s,%s/config/%s' % (BFD_PATH, remote_address, interface, vrf, local_address, attr)
    request = {'path': url, 'method': DELETE}
    return request