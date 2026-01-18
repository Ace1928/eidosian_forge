from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_custom_roles(module):
    return re.findall('^role name (\\S+)', get_config(module, flags=["| include '^role name'"]), re.M)