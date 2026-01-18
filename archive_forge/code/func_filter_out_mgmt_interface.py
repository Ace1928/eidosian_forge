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
def filter_out_mgmt_interface(self, want, have):
    if want:
        mgmt_intf = next((intf for intf in want if intf['name'] == 'Management0'), None)
        if mgmt_intf:
            self._module.fail_json(msg='Management interface should not be configured.')
    for intf in have:
        if intf['name'] == 'Management0':
            have.remove(intf)
            break