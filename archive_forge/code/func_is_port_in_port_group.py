from __future__ import absolute_import, division, print_function
from natsort import (
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.interfaces_util import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import re
import traceback
def is_port_in_port_group(self, intf_name):
    global port_group_interfaces
    if port_group_interfaces is None:
        port_group_interfaces = retrieve_port_group_interfaces(self._module)
    port_num = re.search(port_num_regex, intf_name)
    port_num = int(port_num.group(0))
    if port_num in port_group_interfaces:
        return True
    return False