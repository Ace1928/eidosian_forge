from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def persist_cidr_ipv4(self):
    if self._values['persist_cidr_ipv4'] is None:
        return None
    if 0 <= self._values['persist_cidr_ipv4'] <= 4294967295:
        return self._values['persist_cidr_ipv4']
    raise F5ModuleError("Valid 'persist_cidr_ipv4' must be in range 0 - 4294967295.")