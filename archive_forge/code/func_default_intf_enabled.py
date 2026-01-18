from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def default_intf_enabled(name='', sysdefs=None, mode=None):
    """Get device/version/interface-specific default 'enabled' state.
    L3:
     - Most L3 intfs default to 'shutdown'. Loopbacks default to 'no shutdown'.
     - Some legacy platforms default L3 intfs to 'no shutdown'.
    L2:
     - User-System-Default 'system default switchport shutdown' defines the
       enabled state for L2 intf's. USD defaults may be different on some platforms.
     - An intf may be explicitly defined as L2 with 'switchport' or it may be
       implicitly defined as L2 when USD 'system default switchport' is defined.
    """
    if not name:
        return None
    if sysdefs is None:
        sysdefs = {}
    default = False
    if re.search('port-channel|loopback', name):
        default = True
    elif re.search('Vlan', name):
        default = False
    else:
        if mode is None:
            mode = sysdefs.get('mode')
        if mode == 'layer3':
            default = sysdefs.get('L3_enabled')
        elif mode == 'layer2':
            default = sysdefs.get('L2_enabled')
    return default