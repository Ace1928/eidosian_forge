from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module_base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def addcmd(self, data, tmplt, negate=False):
    """addcmd"""
    command = self._tmplt.render(data, tmplt, negate)
    if command:
        self.commands.extend(to_list(command))