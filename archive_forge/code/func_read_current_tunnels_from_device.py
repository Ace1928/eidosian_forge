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
def read_current_tunnels_from_device(self, tunnel_type):
    uri = 'https://{0}:{1}/mgmt/tm/net/tunnels/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], tunnel_type)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        if 'items' not in response:
            return []
        return [x['fullPath'] for x in response['items']]
    raise F5ModuleError(resp.content)