from __future__ import absolute_import, division, print_function
import os
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip, validate_ip_v6_address
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def ip_encapsulation(self):
    result = cmp_str_with_none(self.want.ip_encapsulation, self.have.ip_encapsulation)
    if result is None:
        return None
    if result == 'inherit':
        return dict(inherit_profile='enabled', ip_encapsulation=[])
    elif result in ['', 'none']:
        return dict(inherit_profile='disabled', ip_encapsulation=[])
    else:
        return dict(inherit_profile='disabled', ip_encapsulation=[dict(name=os.path.basename(result).strip('/'), partition=os.path.dirname(result))])