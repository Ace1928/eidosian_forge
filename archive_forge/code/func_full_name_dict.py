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
def full_name_dict(self):
    if self._values['name'] is None:
        if self._values['address'] and (not self._values['fqdn']):
            name = self._values['address']
        else:
            name = self._values['fqdn']
    else:
        name = self._values['name']
    return dict(name=name, port=self.port)