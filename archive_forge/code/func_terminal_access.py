from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def terminal_access(self):
    if self._values['terminal_access'] in [None, 'tmsh']:
        return self._values['terminal_access']
    elif self._values['terminal_access'] == 'disabled':
        return 'none'
    return self._values['terminal_access']