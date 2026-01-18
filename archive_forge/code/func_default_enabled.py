from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def default_enabled(self, want=None, have=None, action=''):
    if self.state == 'rendered':
        return False
    if want is None:
        want = {}
    if have is None:
        have = {}
    name = have.get('name')
    if name is None:
        return None
    sysdefs = self.intf_defs['sysdefs']
    sysdef_mode = sysdefs['mode']
    intf_def_enabled = self.intf_defs.get(name)
    have_mode = have.get('mode', sysdef_mode)
    if action == 'delete' and (not want):
        want_mode = sysdef_mode
    else:
        want_mode = want.get('mode', have_mode)
    if (want_mode and have_mode) is None or want_mode != have_mode or intf_def_enabled is None:
        intf_def_enabled = default_intf_enabled(name=name, sysdefs=sysdefs, mode=want_mode)
    return intf_def_enabled