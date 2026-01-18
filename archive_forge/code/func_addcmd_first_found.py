from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module_base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def addcmd_first_found(self, data, tmplts, negate=False):
    """addcmd first found"""
    for pname in tmplts:
        before = len(self.commands)
        self.addcmd(data, pname, negate)
        if len(self.commands) != before:
            break