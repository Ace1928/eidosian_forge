from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l3_interfaces.l3_interfaces import L3_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def is_loop_back_already_esist(self, loop_back):
    return ',{0},'.format(loop_back) in self.loop_backs