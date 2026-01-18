from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_delete_bfd_requests(self, commands, have, is_delete_all):
    requests = []
    if not commands:
        return requests
    if is_delete_all:
        requests.extend(self.get_delete_all_bfd_cfg_requests(commands))
    else:
        requests.extend(self.get_delete_bfd_profile_requests(commands, have))
        requests.extend(self.get_delete_bfd_shop_requests(commands, have))
        requests.extend(self.get_delete_bfd_mhop_requests(commands, have))
    return requests