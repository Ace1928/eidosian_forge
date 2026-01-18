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
def dns_resolver(self):
    if self.want.dns_resolver is None:
        return None
    if self.want.dns_resolver == '':
        if self.have.dns_resolver is None or self.have.dns_resolver == 'none':
            return None
        elif self.have.proxy_type == 'explicit' and self.want.proxy_type is None:
            raise F5ModuleError("DNS resolver cannot be empty or 'none' if an existing profile proxy type is set to {0}.".format(self.have.proxy_type))
        elif self.have.dns_resolver is not None:
            return self.want.dns_resolver
    if self.have.dns_resolver is None:
        return self.want.dns_resolver