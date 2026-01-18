from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def enable_dns_express(self):
    if self._values['enable_dns_express'] is None:
        return None
    if self._values['enable_dns_express']:
        return 'yes'
    return 'no'