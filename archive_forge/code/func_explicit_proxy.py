from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def explicit_proxy(self):
    result = dict()
    if self._values['dns_resolver'] is not None:
        result['dnsResolver'] = self._values['dns_resolver']
    if self._values['dns_resolver_address'] is not None:
        result['dnsResolverReference'] = self._values['dns_resolver_address']
    if not result:
        return None
    return result