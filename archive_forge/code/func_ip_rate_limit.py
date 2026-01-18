from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ip_rate_limit(self):
    if self._values['ip_intelligence'] is None:
        return None
    return self._validate_rate_limit(self._values['ip_intelligence']['rate_limit'])