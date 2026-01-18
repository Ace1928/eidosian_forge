from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def dos_protection(self):
    to_filter = dict(dns_publisher=self._values['dns_publisher'], sip_publisher=self._values['sip_publisher'], network_publisher=self._values['network_publisher'])
    result = self._filter_params(to_filter)
    return result