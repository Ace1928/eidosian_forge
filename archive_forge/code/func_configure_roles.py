from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def configure_roles():
    if want['roles']:
        if have:
            for item in set(have['roles']).difference(want['roles']):
                remove('role %s' % item)
            for item in set(want['roles']).difference(have['roles']):
                add('role %s' % item)
        else:
            for item in want['roles']:
                add('role %s' % item)
        return True
    return False