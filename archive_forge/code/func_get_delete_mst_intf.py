from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_mst_intf(self, mst_id, intf_name):
    url = '%s/mstp/mst-instances/mst-instance=%s/interfaces/interface=%s' % (STP_PATH, mst_id, intf_name)
    request = {'path': url, 'method': DELETE}
    return request