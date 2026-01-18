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
def remove_node_from_device(self):
    uri = 'https://{0}:{1}/mgmt/tm/ltm/node/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.node_name))
    response = self.client.api.delete(uri)
    if response.status == 200:
        return True
    raise F5ModuleError(response.content)