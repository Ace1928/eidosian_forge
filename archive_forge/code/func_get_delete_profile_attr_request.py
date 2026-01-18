from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_delete_profile_attr_request(self, profile_name, attr):
    url = '%s/openconfig-bfd-ext:bfd-profile/profile=%s/config/%s' % (BFD_PATH, profile_name, attr)
    request = {'path': url, 'method': DELETE}
    return request