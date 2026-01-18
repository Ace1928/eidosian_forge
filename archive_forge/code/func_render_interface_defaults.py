from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def render_interface_defaults(self, config, intfs):
    """Collect user-defined-default states for 'system default switchport'
        configurations. These configurations determine default L2/L3 modes
        and enabled/shutdown states. The default values for user-defined-default
        configurations may be different for legacy platforms.
        Notes:
        - L3 enabled default state is False on N9K,N7K but True for N3K,N6K
        - Changing L2-L3 modes may change the default enabled value.
        - '(no) system default switchport shutdown' only applies to L2 interfaces.
        Run through the gathered interfaces and tag their default enabled state.
        """
    intf_defs = {}
    L3_enabled = True if re.search('N[356]K', self.get_platform()) else False
    intf_defs = {'sysdefs': {'mode': None, 'L2_enabled': None, 'L3_enabled': L3_enabled}}
    pat = '(no )*system default switchport$'
    m = re.search(pat, config, re.MULTILINE)
    if m:
        intf_defs['sysdefs']['mode'] = 'layer3' if 'no ' in m.groups() else 'layer2'
    pat = '(no )*system default switchport shutdown$'
    m = re.search(pat, config, re.MULTILINE)
    if m:
        intf_defs['sysdefs']['L2_enabled'] = True if 'no ' in m.groups() else False
    for item in intfs:
        intf_defs[item['name']] = default_intf_enabled(name=item['name'], sysdefs=intf_defs['sysdefs'], mode=item.get('mode'))
    return intf_defs